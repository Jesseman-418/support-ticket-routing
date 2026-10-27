"""Support ticket intent classifier with calibrated abstention behavior."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

ABSTAIN_AT = .45

def load_clinc() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the public CLINC OOS benchmark; includes an out-of-scope intent."""
    from datasets import load_dataset
    data = load_dataset("clinc_oos", "plus")
    def convert(split):
        d = data[split].to_pandas()
        return d.rename(columns={"text":"text", "intent":"label"})[["text","label"]]
    return convert("train"), convert("validation"), convert("test")

def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])

def route(model: Pipeline, text: str, threshold: float = ABSTAIN_AT) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("ticket text must not be empty")
    probabilities = model.predict_proba([text])[0]
    index = int(np.argmax(probabilities)); confidence = float(probabilities[index])
    label = str(model.classes_[index])
    return {"category": label if confidence >= threshold else None,
            "confidence": round(confidence, 6), "needs_human_review": confidence < threshold,
            "threshold": threshold}

def evaluate(model: Pipeline, data: pd.DataFrame, threshold: float = ABSTAIN_AT) -> dict[str, Any]:
    y = data.label.astype(str).to_numpy(); probs = model.predict_proba(data.text.astype(str))
    pred = model.classes_[np.argmax(probs, axis=1)].astype(str); confidence = probs.max(axis=1)
    accepted = confidence >= threshold
    return {
        "rows": int(len(data)), "macro_f1_all": float(f1_score(y, pred, average="macro")),
        "macro_f1_per_class": classification_report(y, pred, output_dict=True, zero_division=0),
        "threshold": threshold, "coverage": float(accepted.mean()),
        "accuracy_when_routed": float((pred[accepted] == y[accepted]).mean()) if accepted.any() else None,
        "abstention_rate": float((~accepted).mean()),
    }

def train(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str | Path) -> dict[str, Any]:
    for name, data in [("train",train_df),("validation",validation_df),("test",test_df)]:
        if not {"text","label"}.issubset(data.columns) or data.empty:
            raise ValueError(f"{name} data must have nonempty text and label columns")
    y_train=train_df.label.astype(str); x_train=train_df.text.astype(str)
    baseline=DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train),1)),y_train)
    baseline_pred=baseline.predict(np.zeros((len(validation_df),1)))
    model=build_model().fit(x_train,y_train)
    report={"baseline_validation_macro_f1":float(f1_score(validation_df.label.astype(str),baseline_pred,average="macro")),
            "validation":evaluate(model,validation_df),"test":evaluate(model,test_df),
            "label_count":int(y_train.nunique()),"abstention_threshold":ABSTAIN_AT,
            "data_note":"Use dataset-provided train, validation, and test splits; do not tune on test."}
    output=Path(out_dir); output.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,output/"router.joblib")
    (output/"metrics.json").write_text(json.dumps(report,indent=2))
    return report

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",default="artifacts"); args=parser.parse_args()
    print(json.dumps(train(*load_clinc(),args.output),indent=2))

if __name__=="__main__": main()
