from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

MODEL_PATH = Path("models/blaze_face_short_range.tflite")
BOX_COLOR = (0, 255, 0)  # green in BGR, the channel order OpenCV uses
BOX_THICKNESS = 2

# Detection runs in two stages.
#
# Stage 1 (candidates): scan the full image plus its top and bottom halves
# with a low confidence threshold. Scanning the halves makes small faces
# appear larger to the short-range model, and the low threshold keeps
# borderline faces that would otherwise be missed.
#
# Stage 2 (verification): zoom into every candidate and run the detector
# again on the enlarged crop. A real face scores high when zoomed in;
# a false positive (e.g. blurred crowd texture) does not.
CANDIDATE_CONFIDENCE = 0.3
VERIFICATION_CONFIDENCE = 0.65
DUPLICATE_IOU = 0.4
VERIFICATION_CROP_SIZE = 256

# A face bounding box in pixel coordinates: (x, y, width, height).
FaceBox = tuple[int, int, int, int]


class DetectedFace(NamedTuple):
    """A detected face: its pixel box and the verification confidence."""

    box: FaceBox
    score: float


@lru_cache(maxsize=2)
def _load_face_detector(min_confidence: float) -> FaceDetector:
    """Create a MediaPipe face detector once per threshold and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face detection model not found: {MODEL_PATH}")

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        min_detection_confidence=min_confidence,
    )
    return FaceDetector.create_from_options(options)


def _run_detector(
    image: np.ndarray, min_confidence: float
) -> list[tuple[FaceBox, float]]:
    """Run MediaPipe on a BGR image and return (box, score) pairs."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = _load_face_detector(min_confidence).detect(mp_image)

    detections = []
    for detection in result.detections:
        box = detection.bounding_box
        score = detection.categories[0].score
        detections.append(
            ((box.origin_x, box.origin_y, box.width, box.height), score)
        )
    return detections


def _intersection_over_union(box_a: FaceBox, box_b: FaceBox) -> float:
    """Measure how much two boxes overlap (0 = none, 1 = identical)."""
    overlap_width = max(
        0,
        min(box_a[0] + box_a[2], box_b[0] + box_b[2]) - max(box_a[0], box_b[0]),
    )
    overlap_height = max(
        0,
        min(box_a[1] + box_a[3], box_b[1] + box_b[3]) - max(box_a[1], box_b[1]),
    )
    overlap_area = overlap_width * overlap_height
    union_area = box_a[2] * box_a[3] + box_b[2] * box_b[3] - overlap_area
    return overlap_area / union_area


def _remove_duplicate_faces(faces: list[DetectedFace]) -> list[DetectedFace]:
    """Keep only the highest-scoring face when several boxes overlap."""
    faces = sorted(faces, key=lambda face: face.score, reverse=True)

    kept: list[DetectedFace] = []
    for face in faces:
        overlaps_kept_face = any(
            _intersection_over_union(face.box, kept_face.box) >= DUPLICATE_IOU
            for kept_face in kept
        )
        if not overlaps_kept_face:
            kept.append(face)
    return kept


def _find_candidates(image: np.ndarray) -> list[FaceBox]:
    """Stage 1: collect possible face boxes from the image and its halves."""
    height = image.shape[0]
    regions = [
        (0, image),
        (0, image[: int(height * 0.6)]),
        (int(height * 0.4), image[int(height * 0.4):]),
    ]

    candidates = []
    for offset_y, region in regions:
        for (x, y, width, height), score in _run_detector(
            region, CANDIDATE_CONFIDENCE
        ):
            candidates.append(DetectedFace((x, y + offset_y, width, height), score))

    return [candidate.box for candidate in _remove_duplicate_faces(candidates)]


def _verify_candidate(
    image: np.ndarray, candidate: FaceBox
) -> tuple[float, FaceBox | None]:
    """Stage 2: re-detect inside a zoomed crop around the candidate.

    Returns the best score and the refined face box, or (0.0, None) when
    no face is found inside the candidate area.
    """
    x, y, width, height = candidate
    margin = width // 2
    image_height, image_width = image.shape[:2]

    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image_width, x + width + margin)
    bottom = min(image_height, y + height + margin)

    crop = image[top:bottom, left:right]
    crop_height, crop_width = crop.shape[:2]
    zoomed = cv2.resize(crop, (VERIFICATION_CROP_SIZE, VERIFICATION_CROP_SIZE))

    best_score = 0.0
    best_box: FaceBox | None = None

    for (box_x, box_y, box_w, box_h), score in _run_detector(
        zoomed, VERIFICATION_CONFIDENCE
    ):
        # Map the box from the zoomed crop back to full-image pixels.
        face_x = box_x * crop_width / VERIFICATION_CROP_SIZE + left
        face_y = box_y * crop_height / VERIFICATION_CROP_SIZE + top
        face_width = box_w * crop_width / VERIFICATION_CROP_SIZE
        face_height = box_h * crop_height / VERIFICATION_CROP_SIZE

        # Only accept a face whose center lies inside the candidate box;
        # otherwise a face near the crop edge could wrongly confirm it.
        center_x = face_x + face_width / 2
        center_y = face_y + face_height / 2
        center_is_inside = (
            x <= center_x <= x + width and y <= center_y <= y + height
        )

        if center_is_inside and score > best_score:
            best_score = score
            best_box = (
                int(face_x),
                int(face_y),
                int(face_width),
                int(face_height),
            )

    return best_score, best_box


def detect_faces(image: np.ndarray) -> list[DetectedFace]:
    """Detect faces in a BGR OpenCV image.

    Returns one DetectedFace (pixel box + confidence score) per face,
    or an empty list when no face is found.
    """
    verified_faces = []
    for candidate in _find_candidates(image):
        score, refined_box = _verify_candidate(image, candidate)
        if refined_box is not None:
            verified_faces.append(DetectedFace(refined_box, score))

    return _remove_duplicate_faces(verified_faces)


def draw_bounding_boxes(image: np.ndarray, faces: list[DetectedFace]) -> int:
    """Draw a green rectangle around every detected face.

    The image is modified in place. Returns the number of boxes drawn.
    """
    for (x, y, width, height), _ in faces:
        cv2.rectangle(
            image, (x, y), (x + width, y + height), BOX_COLOR, BOX_THICKNESS
        )

    return len(faces)
