from pathlib import Path
from typing import List, Dict, Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from classifier.preprocessing import TextPreprocessor

MODELS_DIR = Path(__file__).parent.parent / "models"


class DocumentClassifier:
    def __init__(self, algorithm: str = "naive_bayes"):
        self.algorithm = algorithm
        self.pipeline: Optional[Pipeline] = None
        self.categories: List[str] = []

    def _build_pipeline(self, n_samples: int = 100) -> Pipeline:
        min_df = 1 if n_samples < 200 else 2

        if self.algorithm == "svm":
            clf = CalibratedClassifierCV(LinearSVC(max_iter=10000), cv=3)
        else:
            clf = ComplementNB()

        return Pipeline([
            ("preprocessor", TextPreprocessor()),
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=min_df,
            )),
            ("classifier", clf),
        ])

    def train(self, texts: List[str], labels: List[str], test_size: float = 0.2) -> dict:
        self.categories = sorted(set(labels))

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        self.pipeline = self._build_pipeline(n_samples=len(X_train))
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "algorithm": self.algorithm,
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_categories": len(self.categories),
            "categories": self.categories,
        }

    def predict(self, texts: List[str]) -> List[Dict]:
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Run train() or load() first.")

        probas = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_
        results = []

        for i, text in enumerate(texts):
            scores = dict(zip(classes, probas[i].tolist()))
            predicted = classes[np.argmax(probas[i])]
            preview = text[:100].replace("\n", " ").strip()

            results.append({
                "text_preview": preview,
                "predicted_category": predicted,
                "confidence": float(np.max(probas[i])),
                "all_scores": scores,
            })

        return results

    def evaluate(self, texts: List[str], labels: List[str]) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Run train() or load() first.")

        y_pred = self.pipeline.predict(texts)
        report = classification_report(labels, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, y_pred, labels=self.categories)

        return {
            "report": report,
            "confusion_matrix": cm.tolist(),
            "categories": self.categories,
        }

    def save(self, name: str = "default"):
        if self.pipeline is None:
            raise RuntimeError("No model to save. Train a model first.")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({
            "pipeline": self.pipeline,
            "categories": self.categories,
            "algorithm": self.algorithm,
        }, path)
        return path

    def load(self, name: str = "default"):
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.categories = data["categories"]
        self.algorithm = data["algorithm"]
        return self
