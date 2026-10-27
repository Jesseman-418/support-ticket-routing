# Model card

## Intended use
Suggest an intent queue for support tickets and route low-confidence cases to a person. It is a portfolio benchmark, not an autonomous customer-response system.

## Data
CLINC OOS `plus` configuration, with the dataset-provided train, validation, and test splits. It is a curated intent benchmark, not a representative sample of any company's support traffic.

## Method and evaluation
TF-IDF word unigrams/bigrams plus class-weighted logistic regression, compared with a majority-class dummy baseline. Report macro F1 overall and per class, routing coverage, accepted-case accuracy, and abstention rate. The confidence cutoff is a configurable operating point, not a calibrated probability guarantee.

## Risks and limits
Language, intents, class balance, and error costs differ by organization. Review model errors and privacy requirements before using real customer text. Human review is the default for scores below the configured cutoff.
