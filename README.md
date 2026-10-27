# Support Ticket Routing Model

Classify support text into an intent queue, compare a simple majority baseline with TF-IDF + logistic regression, report macro and per-class metrics, and abstain when confidence is low.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/router.py --output artifacts
uvicorn src.api:app --reload
python -m unittest discover -s tests -v
```

The training script loads the public [CLINC OOS](https://huggingface.co/datasets/clinc_oos) `plus` configuration and preserves its train/validation/test partitions. It writes model and evaluation artifacts locally. `POST /route` returns a category suggestion or `needs_human_review=true`; it never sends replies. See [model card](docs/model_card.md).
