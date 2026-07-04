import csv
import shutil
import time
from datetime import timedelta
from pathlib import Path

import kagglehub

from detect import DetectedFace
from main import SUPPORTED_EXTENSIONS, process_image

DATASET_NAME = "xhlulu/140k-real-and-fake-faces"
OUTPUTS_DIR = Path("outputs")
RESULTS_CSV = Path("results.csv")
RESULTS_HEADER = ["filename", "faces_detected", "status", "processing_time_ms"]
PROGRESS_EVERY_N_IMAGES = 100

# Images matching one of these conditions are copied into review/ for
# manual inspection. They are not necessarily wrong: the program does not
# know the ground truth, so these are only worth a human look.
REVIEW_DIR = Path("review")
REVIEW_CSV = Path("review.csv")
REVIEW_HEADER = ["filename", "reason", "faces_detected", "max_confidence", "min_confidence"]
REVIEW_CATEGORIES = ["no_faces", "low_confidence", "many_faces", "processing_failed"]
LOW_CONFIDENCE_THRESHOLD = 0.70
MANY_FACES_THRESHOLD = 10

# Every flagged image gets a row in review.csv, but at most this many
# images per category are copied into review/ so the folders stay small.
# processing_failed/ is exempt: real errors are always copied.
MAX_REVIEW_SAMPLES_PER_CATEGORY = 100


def find_images_recursively(root: Path) -> list[Path]:
    """Return every supported image below the root folder, sorted by path."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_already_processed() -> set[str]:
    """Read results.csv so every image is processed exactly once,
    even if the evaluation is interrupted and restarted."""
    if not RESULTS_CSV.exists():
        return set()

    with RESULTS_CSV.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the header
        return {row[0] for row in reader if row}


def review_reason(faces: list[DetectedFace]) -> str | None:
    """Return why an image needs manual review, or None if it does not."""
    if len(faces) == 0:
        return "no_faces"
    if any(face.score < LOW_CONFIDENCE_THRESHOLD for face in faces):
        return "low_confidence"
    if len(faces) > MANY_FACES_THRESHOLD:
        return "many_faces"
    return None


def load_flagged_counts() -> dict[str, int]:
    """Count review.csv rows per category, so restarts keep correct totals."""
    counts = {category: 0 for category in REVIEW_CATEGORIES}

    if REVIEW_CSV.exists():
        with REVIEW_CSV.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)  # skip the header
            for row in reader:
                if row and row[1] in counts:
                    counts[row[1]] += 1

    return counts


def count_existing_review_copies() -> dict[str, int]:
    """Count images already copied into review/, so restarts respect the cap."""
    return {
        category: sum(1 for _ in (REVIEW_DIR / category).glob("*"))
        if (REVIEW_DIR / category).is_dir()
        else 0
        for category in REVIEW_CATEGORIES
    }


def add_to_review_queue(
    writer: csv.writer,
    relative_name: str,
    reason: str,
    faces: list[DetectedFace],
    source_path: Path,
    copied_counts: dict[str, int],
) -> None:
    """Record the image in review.csv and copy it into review/<reason>/.

    Copies are capped per category so the folders stay small; only
    processing_failed/ always receives every image.
    """
    scores = [face.score for face in faces]
    max_confidence = f"{max(scores):.2f}" if scores else ""
    min_confidence = f"{min(scores):.2f}" if scores else ""
    writer.writerow([relative_name, reason, len(faces), max_confidence, min_confidence])

    copy_allowed = (
        reason == "processing_failed"
        or copied_counts[reason] < MAX_REVIEW_SAMPLES_PER_CATEGORY
    )
    if not copy_allowed:
        return

    review_folder = REVIEW_DIR / reason
    review_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_path, review_folder / relative_name.replace("/", "_"))
    copied_counts[reason] += 1


def main() -> None:
    dataset_root = Path(kagglehub.dataset_download(DATASET_NAME))
    print(f"Dataset location: {dataset_root}", flush=True)

    image_paths = find_images_recursively(dataset_root)
    total = len(image_paths)

    if total == 0:
        print("Error: no images found in the dataset")
        return

    already_processed = load_already_processed()
    if already_processed:
        print(f"Resuming: {len(already_processed)} images already in {RESULTS_CSV}")

    results_needs_header = not RESULTS_CSV.exists()
    review_needs_header = not REVIEW_CSV.exists()
    flagged_counts = load_flagged_counts()
    copied_counts = count_existing_review_copies()
    processed = len(already_processed)
    failed = 0
    total_faces = 0
    start_time = time.perf_counter()

    with (
        RESULTS_CSV.open("a", newline="") as results_file,
        REVIEW_CSV.open("a", newline="") as review_file,
    ):
        results_writer = csv.writer(results_file)
        review_writer = csv.writer(review_file)
        if results_needs_header:
            results_writer.writerow(RESULTS_HEADER)
        if review_needs_header:
            review_writer.writerow(REVIEW_HEADER)

        for image_path in image_paths:
            relative_name = str(image_path.relative_to(dataset_root))
            if relative_name in already_processed:
                continue

            output_path = OUTPUTS_DIR / relative_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            image_start = time.perf_counter()
            try:
                faces = process_image(image_path, output_path)
                status = "ok"
                total_faces += len(faces)
                reason = review_reason(faces)
                review_source = output_path
            except Exception as error:
                faces = []
                status = "failed"
                failed += 1
                reason = "processing_failed"
                review_source = image_path
                print(f"Warning: {relative_name} failed ({error})")

            elapsed_ms = round((time.perf_counter() - image_start) * 1000)
            results_writer.writerow([relative_name, len(faces), status, elapsed_ms])
            results_file.flush()

            if reason is not None:
                add_to_review_queue(
                    review_writer, relative_name, reason, faces,
                    review_source, copied_counts,
                )
                flagged_counts[reason] += 1
                review_file.flush()

            processed += 1
            if processed % PROGRESS_EVERY_N_IMAGES == 0:
                print(f"Processed: {processed} / {total}", flush=True)

    total_seconds = time.perf_counter() - start_time
    newly_processed = processed - len(already_processed)

    print()
    print(f"Total images processed: {processed}")
    print(f"Failed images: {failed}")
    print(f"Total faces detected: {total_faces}")

    if processed > failed:
        print(f"Average faces per image: {total_faces / (processed - failed):.2f}")

    print(f"Total runtime: {timedelta(seconds=round(total_seconds))}")

    if newly_processed > 0:
        print(
            f"Average runtime per image: "
            f"{total_seconds / newly_processed * 1000:.0f} ms"
        )

    print()
    print("Review samples saved:")
    for category in REVIEW_CATEGORIES:
        print(f"  {category}: {copied_counts[category]} / {flagged_counts[category]}")


if __name__ == "__main__":
    main()
