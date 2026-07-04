from pathlib import Path

import cv2

from detect import DetectedFace, detect_faces, draw_bounding_boxes

DATASET_DIR = Path("dataset")
OUTPUTS_DIR = Path("outputs")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_image_files(dataset_dir: Path) -> list[Path]:
    """Return all supported image files in the dataset folder, sorted by name."""
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def process_image(image_path: Path, output_path: Path) -> list[DetectedFace]:
    """Detect faces in one image, draw boxes, and save the result.

    Returns the detected faces. Raises ValueError when the image cannot
    be read or the result cannot be saved.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("image could not be read")

    faces = detect_faces(image)
    draw_bounding_boxes(image, faces)

    if not cv2.imwrite(str(output_path), image):
        raise ValueError("processed image could not be saved")

    return faces


def main() -> None:
    if not DATASET_DIR.is_dir():
        print(f"Error: dataset folder not found: {DATASET_DIR}")
        return

    image_paths = find_image_files(DATASET_DIR)

    if not image_paths:
        print(f"Error: no images found in {DATASET_DIR}")
        return

    OUTPUTS_DIR.mkdir(exist_ok=True)

    images_processed = 0
    total_faces = 0

    for image_path in image_paths:
        try:
            faces = process_image(image_path, OUTPUTS_DIR / image_path.name)
        except Exception as error:
            print(f"Warning: skipped {image_path.name} ({error})")
            continue

        face_count = len(faces)

        print(f"{image_path.name} -> Faces detected: {face_count}")
        images_processed += 1
        total_faces += face_count

    print()
    print(f"Images processed: {images_processed}")
    print(f"Total faces detected: {total_faces}")

    if images_processed > 0:
        print(f"Average faces per image: {total_faces / images_processed:.2f}")


if __name__ == "__main__":
    main()
