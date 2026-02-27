from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt

from classifier.core import DocumentClassifier
from classifier.dataset import get_demo_dataset
from classifier.display import (
    show_banner,
    show_classification_results,
    show_confusion_matrix,
    show_detailed_result,
    show_error,
    show_evaluation_report,
    show_info,
    show_success,
    show_training_results,
)
from classifier.reader import get_supported_extensions, read_document

app = typer.Typer(
    name="docclassify",
    help="ML-Powered Document Classification Engine",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _load_directory_data(data_dir: Path):
    texts, labels = [], []
    extensions = set(get_supported_extensions())
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        category = category_dir.name
        for file_path in sorted(category_dir.iterdir()):
            if file_path.suffix.lower() in extensions:
                try:
                    text = read_document(file_path)
                    if text.strip():
                        texts.append(text)
                        labels.append(category)
                except Exception:
                    console.print(f"[dim]Skipping {file_path.name}: could not read[/]")
    return texts, labels


@app.command()
def train(
    algorithm: str = typer.Option("naive_bayes", "--algorithm", "-a", help="Algorithm: naive_bayes or svm"),
    model_name: str = typer.Option("default", "--model-name", "-m", help="Name for the saved model"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Directory with categorized documents"),
    demo: bool = typer.Option(False, "--demo", help="Use built-in demo dataset"),
):
    show_banner()

    if not demo and data_dir is None:
        show_error("Provide --demo or --data-dir PATH")
        raise typer.Exit(1)

    if demo:
        show_info("Loading built-in demo dataset (120 documents, 6 categories)")
        texts, labels = get_demo_dataset()
    else:
        if not data_dir.exists():
            show_error(f"Directory not found: {data_dir}")
            raise typer.Exit(1)
        show_info(f"Loading documents from {data_dir}")
        texts, labels = _load_directory_data(data_dir)
        if len(texts) == 0:
            show_error("No documents found in the specified directory")
            raise typer.Exit(1)

    classifier = DocumentClassifier(algorithm=algorithm)

    with console.status("[bold cyan]Training model...[/]", spinner="dots"):
        results = classifier.train(texts, labels)

    show_training_results(results)

    with console.status("[bold cyan]Saving model...[/]", spinner="dots"):
        path = classifier.save(model_name)

    show_success(f"Model saved to {path}")


@app.command()
def classify(
    path: Path = typer.Argument(..., help="File or directory to classify"),
    model_name: str = typer.Option("default", "--model-name", "-m", help="Model name to load"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed scores for each category"),
):
    show_banner()

    classifier = DocumentClassifier()
    try:
        classifier.load(model_name)
    except FileNotFoundError:
        show_error(f"Model '{model_name}' not found. Run 'docclassify train' first.")
        raise typer.Exit(1)

    path = Path(path)
    texts = []
    if path.is_file():
        try:
            texts.append(read_document(path))
        except (ValueError, FileNotFoundError) as e:
            show_error(str(e))
            raise typer.Exit(1)
    elif path.is_dir():
        extensions = set(get_supported_extensions())
        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() in extensions:
                try:
                    texts.append(read_document(file_path))
                except Exception:
                    console.print(f"[dim]Skipping {file_path.name}: could not read[/]")
    else:
        show_error(f"Path not found: {path}")
        raise typer.Exit(1)

    if not texts:
        show_error("No readable documents found")
        raise typer.Exit(1)

    with console.status("[bold cyan]Classifying...[/]", spinner="dots"):
        results = classifier.predict(texts)

    if detailed:
        for result in results:
            show_detailed_result(result)
    else:
        show_classification_results(results)


@app.command()
def evaluate(
    model_name: str = typer.Option("default", "--model-name", "-m", help="Model name to load"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Directory with categorized documents"),
    demo: bool = typer.Option(False, "--demo", help="Use built-in demo dataset"),
):
    show_banner()

    classifier = DocumentClassifier()
    try:
        classifier.load(model_name)
    except FileNotFoundError:
        show_error(f"Model '{model_name}' not found. Run 'docclassify train' first.")
        raise typer.Exit(1)

    if not demo and data_dir is None:
        show_error("Provide --demo or --data-dir PATH")
        raise typer.Exit(1)

    if demo:
        show_info("Evaluating on built-in demo dataset")
        texts, labels = get_demo_dataset()
    else:
        if not data_dir.exists():
            show_error(f"Directory not found: {data_dir}")
            raise typer.Exit(1)
        show_info(f"Loading documents from {data_dir}")
        texts, labels = _load_directory_data(data_dir)
        if len(texts) == 0:
            show_error("No documents found in the specified directory")
            raise typer.Exit(1)

    with console.status("[bold cyan]Evaluating...[/]", spinner="dots"):
        eval_data = classifier.evaluate(texts, labels)

    show_evaluation_report(eval_data)
    console.print()
    show_confusion_matrix(eval_data["confusion_matrix"], eval_data["categories"])


@app.command()
def interactive(
    model_name: str = typer.Option("default", "--model-name", "-m", help="Model name to load"),
):
    show_banner()

    classifier = DocumentClassifier()
    try:
        classifier.load(model_name)
    except FileNotFoundError:
        show_error(f"Model '{model_name}' not found. Run 'docclassify train' first.")
        raise typer.Exit(1)

    show_info("Interactive mode — paste text and press Enter. Type 'quit' to exit.")
    console.print()

    while True:
        try:
            text = Prompt.ask("[bold cyan]Enter text[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if text.lower().strip() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        if not text.strip():
            continue

        results = classifier.predict([text])
        show_detailed_result(results[0])
        console.print()


if __name__ == "__main__":
    app()
