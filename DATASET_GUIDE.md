# Dataset Guide

SoyCare AI trains on real soybean leaf photographs. This guide covers where the
data comes from, how to fetch it, and the checks that must pass before any
retraining is allowed to proceed.

## Two class lists, on purpose

| Constant | Meaning |
|---|---|
| `config.CLASSES` | The taxonomy the **shipped model** was trained on. Currently 3 classes. Inference reads only this list. |
| `config.TARGET_CLASSES` | The 6-class taxonomy the knowledge base documents and Phase 0 is assembling data for. |

`CLASSES` is promoted to `TARGET_CLASSES` only after the dataset audit passes
and a 6-class model has been trained. `DiseasePredictor._verify_class_alignment`
refuses to load a model whose output width does not match `CLASSES`, so the two
can never silently drift apart and mislabel predictions.

Target classes, alphabetical, matching the on-disk folder names:

- `Bacterial blight`
- `Downy mildew`
- `Frogeye leaf spot`
- `Healthy`
- `Septoria brown spot`
- `Soybean rust`

## Sources

`data/sources.json` is the registry of record. Every entry carries its DOI,
license, capture conditions, citation, and the mapping from its native class
names onto `TARGET_CLASSES`. Rejected sources are recorded too, with the reason,
so the same candidates are not re-litigated later.

Current position:

| Source | License | Supplies |
|---|---|---|
| Auburn Soybean Disease Image Dataset (ASDID) | CC0-1.0 | Bacterial blight, Downy mildew, Frogeye leaf spot, Healthy, Soybean rust |
| India soyabean dataset (Project-AgML mirror) | CC-BY-4.0 | Bacterial blight, Septoria brown spot, Healthy |

**Known gap.** Septoria brown spot has roughly 284 images from a single Indian
source, below the 300 minimum, and no US field-conditions source. This is the
blocking gap for the six-class retrain. Closing it needs either a second
published source or your own field photography.

## Fetch

```bash
# Inspect the registry
python -m scripts.fetch_datasets --list

# Pull one source into data/raw/<class>/ with a provenance manifest
python -m scripts.fetch_datasets --source asdid
python -m scripts.fetch_datasets --source agml_india_soyabean

# Limit scope while iterating
python -m scripts.fetch_datasets --source asdid --classes "Frogeye leaf spot" --max-per-class 50
```

The ASDID archives are 2.8-8.4 GB each. They are served with HTTP range
support, so the fetcher reads each zip's central directory and pulls only the
image members it needs rather than downloading whole archives. Images are
downscaled to 1024 px on the longest edge by default (`--max-dimension 0` keeps
originals), because the raw photographs are 5472x3648 at roughly 6 MB each while
the model only ever sees 224x224.

Every fetched image is recorded in `data/raw/_provenance.jsonl` with its source
id, DOI, license, native class, and byte size.

## Audit

```bash
python -m scripts.audit_dataset
```

Exits non-zero unless every class has at least `config.MIN_IMAGES_PER_CLASS`
(300) readable images. It reports per class: counts, files that do not decode,
exact duplicates by SHA-256, near-duplicate clusters, and resolution spread. The
full report is written to `outputs/dataset_audit.json`.

An absent class fails the gate, not just an under-filled one. A six-class model
trained on a silently empty class is the failure this gate exists to prevent.

## Split

```bash
# Stage the six-class corpus before CLASSES has been promoted
python -m src.prepare_dataset --raw-dir data/raw \
  --classes "Bacterial blight,Downy mildew,Frogeye leaf spot,Healthy,Septoria brown spot,Soybean rust"
```

Splits are assigned per **capture-session group**, not per image. Public dataset
filenames carry no session metadata, so `src/dataset_groups.py` approximates
sessions by clustering perceptually near-identical images with a difference
hash; a burst of frames of one leaf collapses into one group, and whole groups
are assigned to a single split. This is what `DATASET_GUIDE` has always
required, and the previous per-image shuffle violated it, which inflates
measured accuracy.

## Label quality

- Label only images confirmed by a trusted source, agricultural specialist, or dataset documentation.
- Do not label a leaf as rust from appearance alone when symptoms are ambiguous.
- Keep uncertain images outside the training set until reviewed.
- Record the source and license for each dataset used.
- Do not mix images from the same capture session across train, validation, and test.

## Prepare and train

From the repository root:

```bash
rm -rf data/processed
python -m src.prepare_dataset --raw-dir data/raw
python -m src.train --epochs1 15 --epochs2 10
python -m src.evaluate
```

Review `outputs/evaluation_report.json` and the confusion matrix. Do not use the
model for real decisions unless performance is measured on a held-out real test
set.

## Acceptance checks

Pay special attention to:

- Soybean rust recall: missed rust cases are high risk.
- Healthy precision: false healthy predictions are unsafe.
- Per-class confusion, not only overall accuracy.
- Performance on photos from a farm or camera not present in training.

A model that performs well only on one source or background is not ready for
deployment. Mixed-source corpora make this concrete: prefer a test split drawn
from a different source and region than the training split, so the reported
number reflects field generalisation rather than source memorisation.
