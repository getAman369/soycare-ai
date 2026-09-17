# Real Dataset Guide

The current development model uses real soybean leaf photographs for three
classes. Expand to more diseases only when matching real labeled data is
available.

## Current experiment classes

Use these exact directory names:

- `Frogeye leaf spot`
- `Healthy`
- `Soybean rust`

Place images under `data/raw/<class name>/`.

## Collection target

Aim for at least 300 real images per class for an initial model, with different:

- fields, farms, cultivars, and growth stages
- lighting, backgrounds, camera distances, and viewing angles
- disease severity levels
- healthy and visibly affected leaves

More important than the exact count is that photos from the same plant or near-duplicate frames stay in the same split.

## Label quality

- Label only images confirmed by a trusted source, agricultural specialist, or dataset documentation.
- Do not label a leaf as rust from appearance alone when the symptoms are ambiguous.
- Keep uncertain images outside the training set until reviewed.
- Record the source and license for each dataset used.
- Do not mix images from the same capture session across train, validation, and test.

## Prepare and train

From the `soyabean` directory:

```bash
rm -rf data/processed
.venv/bin/python -m src.prepare_dataset --raw-dir data/raw
.venv/bin/python -m src.train --epochs1 15 --epochs2 10
.venv/bin/python -m src.evaluate
```

Review `outputs/evaluation_report.json` and the confusion matrix. Do not use the model for real decisions unless performance is measured on a held-out real test set.

## Acceptance checks

Pay special attention to:

- Soybean rust recall: missed rust cases are high risk.
- Healthy precision: false healthy predictions are unsafe.
- Per-class confusion, not only overall accuracy.
- Performance on photos from a farm or camera not present in training.

A model that performs well only on one source or background is not ready for deployment.
