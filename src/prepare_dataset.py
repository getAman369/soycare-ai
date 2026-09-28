"""
Dataset Preparation & Split Utility
Organises real image folders into train/validation/test splits and provides a
synthetic sample generator to test the training pipeline immediately.

Splits are assigned per capture-session group rather than per image, so frames
photographed of the same plant cannot land in different splits and inflate the
measured accuracy.
"""
from pathlib import Path
import random
import shutil
import argparse
import logging
from PIL import Image, ImageDraw, ImageFilter

from config import DATA_DIR, CLASSES, IMAGE_SIZE
from src.dataset_groups import assign_groups_to_splits, group_paths

logger = logging.getLogger("soycare.dataset")

DEFAULT_SEED = 20240501


def setup_folder_structure(classes=None):
    """Creates the class directories for train, validation, and test splits."""
    classes = classes or CLASSES
    processed_dir = DATA_DIR / "processed"
    for split in ["train", "validation", "test"]:
        for class_name in classes:
            folder = processed_dir / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
    logger.info("Created folder structure in %s for all %d classes.", processed_dir, len(classes))


def split_raw_dataset(
    raw_dir=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=DEFAULT_SEED,
    classes=None,
):
    """
    Takes organised folders from data/raw/<class_name>/*.jpg and copies them into
    data/processed/{train,validation,test}/<class_name>.

    Images are clustered into capture-session groups by perceptual hash and whole
    groups are assigned to a split, which keeps near-duplicate frames of the same
    leaf on one side of the split boundary.

    `classes` defaults to config.CLASSES, the taxonomy the deployed model was
    trained on. Pass config.TARGET_CLASSES explicitly to stage the six-class
    corpus before the retrain promotes TARGET_CLASSES to CLASSES.
    """
    if raw_dir is None:
        raw_dir = DATA_DIR / "raw"

    classes = classes or CLASSES
    raw_dir = Path(raw_dir)
    processed_dir = DATA_DIR / "processed"

    if not raw_dir.exists():
        logger.error("Raw directory %s does not exist.", raw_dir)
        return None

    setup_folder_structure(classes)

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ratios = {"train": train_ratio, "validation": val_ratio, "test": test_ratio}
    total_moved = 0
    summary = {}

    for class_name in classes:
        class_raw = raw_dir / class_name
        if not class_raw.exists():
            continue

        images = [f for f in class_raw.iterdir() if f.suffix.lower() in valid_extensions]
        if not images:
            continue

        groups = group_paths(images)
        assignment = assign_groups_to_splits(groups, ratios, seed=seed)

        per_split = {split: [] for split in ratios}
        for group_id, split in assignment.items():
            per_split[split].extend(groups[group_id])

        for split_name, files in per_split.items():
            dest_dir = processed_dir / split_name / class_name
            for file_path in files:
                shutil.copy2(file_path, dest_dir / file_path.name)

        summary[class_name] = {split: len(files) for split, files in per_split.items()}
        total_moved += len(images)
        logger.info(
            "Class '%s': %d images in %d groups -> %s",
            class_name, len(images), len(groups),
            ", ".join(f"{split}={len(files)}" for split, files in per_split.items()),
        )

    logger.info("Completed dataset splitting! Total images organised: %d", total_moved)
    return summary


def generate_sample_dataset(samples_per_class=30):
    """
    Generates synthetic leaf images with simulated disease lesion patterns
    allowing developers to test the full EfficientNetB0 training pipeline immediately.
    """
    setup_folder_structure()
    processed_dir = DATA_DIR / "processed"

    # Color palettes for different disease symptoms
    palettes = {
        "Healthy": [(34, 139, 34), (46, 125, 50), (76, 175, 80)],
        "Bacterial blight": [(34, 139, 34), (139, 69, 19), (200, 180, 50)],
        "Downy mildew": [(46, 125, 50), (220, 220, 100), (180, 180, 180)],
        "Frogeye leaf spot": [(34, 139, 34), (120, 50, 20), (210, 180, 140)],
        "Septoria brown spot": [(46, 125, 50), (90, 50, 20), (160, 82, 45)],
        "Soybean rust": [(34, 139, 34), (184, 115, 51), (139, 0, 0)]
    }

    logger.info("Generating %d sample images per class for training validation...", samples_per_class)

    for class_name in CLASSES:
        colors = palettes.get(class_name, [(34, 139, 34), (100, 100, 100)])

        train_count = int(samples_per_class * 0.7)
        val_count = int(samples_per_class * 0.15)
        test_count = samples_per_class - train_count - val_count

        splits = [
            ("train", train_count),
            ("validation", val_count),
            ("test", test_count)
        ]

        for split_name, count in splits:
            folder = processed_dir / split_name / class_name
            for i in range(count):
                # Create leaf base
                base_color = colors[0]
                img = Image.new("RGB", IMAGE_SIZE, color=(base_color[0] + random.randint(-15, 15),
                                                          base_color[1] + random.randint(-15, 15),
                                                          base_color[2] + random.randint(-10, 10)))
                draw = ImageDraw.Draw(img)

                # Draw vein structure
                draw.line([(112, 10), (112, 214)], fill=(20, 90, 20), width=3)
                for y in range(40, 200, 30):
                    draw.line([(112, y), (40, y + 25)], fill=(25, 100, 25), width=2)
                    draw.line([(112, y), (184, y + 25)], fill=(25, 100, 25), width=2)

                # Draw disease specific lesion patterns if not healthy
                if class_name != "Healthy":
                    lesion_color = colors[1]
                    halo_color = colors[2] if len(colors) > 2 else colors[1]

                    for _ in range(random.randint(6, 18)):
                        cx = random.randint(35, 185)
                        cy = random.randint(35, 185)
                        r = random.randint(5, 16)

                        # Draw halo
                        draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill=halo_color)
                        # Draw lesion core
                        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=lesion_color)

                img = img.filter(ImageFilter.SMOOTH_MORE)
                img.save(folder / f"sample_{class_name.lower().replace(' ', '_')}_{split_name}_{i:03d}.jpg", quality=90)

    logger.info("Sample dataset created successfully! Ready for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoyCare AI Dataset Preparation")
    parser.add_argument("--generate-samples", action="store_true", help="Generate synthetic samples to test training immediately")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to raw dataset directory with class folders")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for the reproducible group split")
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class names to split. Defaults to config.CLASSES; "
             "use config.TARGET_CLASSES to stage the six-class corpus.",
    )
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None

    if args.generate_samples:
        generate_sample_dataset()
    elif args.raw_dir:
        split_raw_dataset(args.raw_dir, seed=args.seed, classes=classes)
    else:
        setup_folder_structure()
        print("Folder structure initialized. Use --generate-samples to create sample data or --raw-dir to split raw images.")
