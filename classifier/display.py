import io
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

_utf8_out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
console = Console(file=_utf8_out, force_terminal=True)


def show_banner():
    banner = Text()
    banner.append("DOCUMENT CLASSIFIER", style="bold cyan")
    banner.append("\n")
    banner.append("ML-Powered Document Classification Engine", style="dim white")
    banner.append("\n")
    banner.append("TF-IDF + Naive Bayes / SVM Pipeline", style="dim cyan")
    console.print(Panel(banner, box=box.DOUBLE, border_style="cyan", padding=(1, 4)))


def _confidence_bar(value: float, width: int = 20) -> Text:
    filled = int(value * width)
    empty = width - filled
    if value >= 0.7:
        color = "green"
    elif value >= 0.4:
        color = "yellow"
    else:
        color = "red"
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style="dim")
    bar.append(f" {value:.1%}", style=f"bold {color}")
    return bar


def show_training_results(results: dict):
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan")
    table.add_row("Algorithm", results.get("algorithm", "N/A"))
    table.add_row("Training Samples", str(results.get("train_size", "N/A")))
    table.add_row("Test Samples", str(results.get("test_size", "N/A")))
    table.add_row("Categories", str(results.get("num_categories", "N/A")))
    accuracy = results.get("accuracy", 0)
    bar = _confidence_bar(accuracy, width=30)
    accuracy_text = Text()
    accuracy_text.append_text(bar)
    table.add_row("Accuracy", accuracy_text)
    console.print(Panel(table, title="[bold green]Training Complete[/]", border_style="green", box=box.ROUNDED))


def show_classification_results(results: list):
    table = Table(box=box.ROUNDED, border_style="blue", title="Classification Results", title_style="bold blue")
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Document", style="white", max_width=40)
    table.add_column("Category", style="bold cyan", width=16)
    table.add_column("Confidence", width=30)
    for i, result in enumerate(results, 1):
        bar = _confidence_bar(result["confidence"])
        table.add_row(str(i), result["text_preview"], result["predicted_category"].upper(), bar)
    console.print(table)


def show_detailed_result(result: dict):
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Category", style="bold white", width=16)
    table.add_column("Score", width=35)
    sorted_scores = sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True)
    for category, score in sorted_scores:
        bar = _confidence_bar(score, width=25)
        table.add_row(category.upper(), bar)
    preview = result["text_preview"][:80] + "..." if len(result["text_preview"]) > 80 else result["text_preview"]
    panel_title = f"[bold cyan]{result['predicted_category'].upper()}[/] — {preview}"
    console.print(Panel(table, title=panel_title, border_style="cyan", box=box.ROUNDED))


def show_evaluation_report(eval_data: dict):
    report = eval_data["report"]
    table = Table(
        box=box.ROUNDED,
        border_style="magenta",
        title="Evaluation Report",
        title_style="bold magenta",
    )
    table.add_column("Category", style="bold white", width=16)
    table.add_column("Precision", justify="right", width=10)
    table.add_column("Recall", justify="right", width=10)
    table.add_column("F1-Score", justify="right", width=10)
    table.add_column("Support", justify="right", width=10)

    for label in sorted(report.keys()):
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        metrics = report[label]
        table.add_row(
            label.upper(),
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}",
            str(int(metrics["support"])),
        )

    table.add_section()
    for avg_type in ("macro avg", "weighted avg"):
        if avg_type in report:
            metrics = report[avg_type]
            table.add_row(
                avg_type.upper(),
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                str(int(metrics["support"])),
                style="bold",
            )

    console.print(table)


def show_confusion_matrix(cm, categories: list):
    table = Table(
        box=box.ROUNDED,
        border_style="yellow",
        title="Confusion Matrix",
        title_style="bold yellow",
    )
    table.add_column("", style="bold white", width=14)
    for cat in categories:
        table.add_column(cat[:6].upper(), justify="center", width=8)

    for i, row_cat in enumerate(categories):
        cells = []
        for j, val in enumerate(cm[i]):
            if i == j:
                cells.append(f"[bold green]{val}[/]")
            elif val > 0:
                cells.append(f"[bold red]{val}[/]")
            else:
                cells.append(f"[dim]{val}[/]")
        table.add_row(row_cat[:14].upper(), *cells)

    console.print(table)


def show_error(message: str):
    console.print(Panel(f"[bold red]{message}[/]", border_style="red", box=box.ROUNDED, title="[red]Error[/]"))


def show_success(message: str):
    console.print(Panel(f"[bold green]{message}[/]", border_style="green", box=box.ROUNDED, title="[green]Success[/]"))


def show_info(message: str):
    console.print(Panel(f"[bold blue]{message}[/]", border_style="blue", box=box.ROUNDED, title="[blue]Info[/]"))
