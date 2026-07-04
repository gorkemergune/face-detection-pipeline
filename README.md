<div align="center">

# Face Detection Pipeline

**Two-stage face detection built on MediaPipe BlazeFace — batch processing, human-review queues, and a 140,000-image CPU benchmark.**

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazeFace-ff6f00)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8)](https://opencv.org/)
[![License](https://img.shields.io/github/license/gorkemergune/face-detection-pipeline)](LICENSE)
[![Benchmark](https://img.shields.io/badge/benchmark-140k%20images%20·%200%20failures-success)](#-performance)
[![CPU](https://img.shields.io/badge/GPU-not%20required-lightgrey)](#-performance)

<img src="img/examples/multi_face_detected.jpg" width="640" alt="Face detection result with bounding boxes">

</div>

---

##  Overview

Off-the-shelf short-range face detectors are fast but fragile: small faces get missed, and blurred textures produce false positives. No single confidence threshold fixes both — raising it loses real faces, lowering it invents fake ones.

This project wraps MediaPipe's BlazeFace model in a **two-stage detect-and-verify pipeline**: candidates are collected with a permissive threshold, then each one is zoomed into and re-detected. Real faces re-score 0.90+ when enlarged; false positives fail re-detection. The result is measurably better accuracy with the same model, the same dependencies, and ~11 ms per image on a laptop CPU.

---

##  Features

-  **Two-stage detection** — low-threshold candidate search, then zoom-in re-verification of every candidate
-  **Batch processing** — process an entire folder; annotated copies are saved automatically
-  **Large-scale evaluation** — 140k-image Kaggle benchmark with per-image CSV logging and safe resume after interruption
-  **Human-review queue** — zero-face, low-confidence, and many-face images are sampled into `review/` folders for manual QA, with complete metadata in `review.csv`
-  **Robust by default** — corrupt or unreadable images are skipped with a warning, never crashing a batch
-  **CPU-only** — no GPU, no cloud, no model download step (the 230 KB model ships with the repo)

##  Demo

| Original | Detected |
|:---:|:---:|
| <img src="img/examples/portrait_original.jpg" width="330"> | <img src="img/examples/portrait_detected.jpg" width="330"> |
| *Single face — studio portrait* | *1 face, confidence 0.91* |
| <img src="img/examples/multi_face_original.jpg" width="330"> | <img src="img/examples/multi_face_detected.jpg" width="330"> |
| *Multiple faces in one frame* | *2 faces detected in a single pass* |
| <img src="img/examples/stadium_night_original.jpg" width="330"> | <img src="img/examples/stadium_night_detected.jpg" width="330"> |
| *Night match — floodlights, blurred crowd* | *Crowd false positives rejected by verification* |
| <img src="img/examples/small_face_original.jpg" width="330"> | <img src="img/examples/small_face_detected.jpg" width="330"> |
| *Full-body shot — face is a tiny part of the frame* | *Recovered by regional candidate search* |
| <img src="img/examples/daylight_crowd_original.jpg" width="330"> | <img src="img/examples/daylight_crowd_detected.jpg" width="330"> |
| *Harsh daylight, tilted face, busy background* | *Missed entirely by single-pass detection — found here* |
| <img src="img/examples/red_jersey_original.jpg" width="330"> | <img src="img/examples/red_jersey_detected.jpg" width="330"> |
| *Noisy stadium background (10+ raw candidates)* | *Exactly one confirmed face* |

##  How It Works

```text
                ┌─────────────────────────────────────────────┐
 input image ─▶ │ Stage 1 · Candidate search                  │
                │  detector on full image + top/bottom halves │
                │  (small faces appear larger), threshold 0.30│
                └──────────────────┬──────────────────────────┘
                                   ▼
                ┌─────────────────────────────────────────────┐
                │ Stage 2 · Verification                      │
                │  zoom into each candidate, re-detect at 0.65│
                │  accept only faces centered in the candidate│
                │  refine the box to the zoomed detection     │
                └──────────────────┬──────────────────────────┘
                                   ▼
                ┌─────────────────────────────────────────────┐
                │ Deduplication · IoU ≥ 0.4 → keep best score │
                └──────────────────┬──────────────────────────┘
                                   ▼
                      bounding boxes + confidence scores
```


##  Results

From the full 140.000 benchmark, only **0.26%** of images were flagged for human review:

| Review category | Flagged | Sampled to `review/` |
|---|---:|---:|
| Low confidence (< 0.70) | 354 | 100 (capped) |
| No faces found | 13 | 13 |
| Many faces (> 10) | 0 | 0 |
| Processing failed | 0 | 0 |

The review queue never claims an image is *wrong* — ground truth is unknown to the program. It simply routes the least-certain outputs to a human, with full confidence metadata in `review.csv`.

##  Performance

Full benchmark on the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset — single-threaded, Apple M3 CPU:

| Metric | Value |
|---|---:|
| Images processed | **140,000** |
| Failed images | **0** |
| Total faces detected | **140,836** |
| Average faces per image | 1.006 |
| Total runtime | **25 m 17 s** |
| Average time per image | **~11 ms** |

<div align="center">
<img src="img/summary.png" width="420" alt="Benchmark run summary">
</div>

## Installation

Requires **Python 3.12+**.

```bash
git clone https://github.com/gorkemergune/face-detection-pipeline.git
cd face-detection-pipeline

python3 -m venv venv
source venv/bin/activate        

pip install -r requirements.txt
```

No extra downloads — the BlazeFace model is bundled in `models/`.

## Usage

**Process your own images** — drop them into `dataset/`, run, and collect annotated copies from `outputs/`:

```bash
python src/main.py
```

```text
IMG_7418.jpg -> Faces detected: 1
IMG_7423.jpg -> Faces detected: 1
IMG_7426.jpg -> Faces detected: 1

Images processed: 10
Total faces detected: 10
Average faces per image: 1.00
```

**Run the 140k benchmark** — downloads the dataset (~3.8 GB) via `kagglehub`, writes `results.csv`, `review.csv`, and the `review/` sample folders:

```bash
python src/evaluate.py
```

Interrupt it anytime — a restart resumes exactly where it stopped, never processing an image twice.

> Both scripts are run from the repository root (paths are relative to it).

## Project Structure

```text
face-detection-pipeline/
├── dataset/            # Input images (you supply these; not committed)
├── docs/               # Development docs: roadmap, task list, coding rules
├── img/                # README assets (examples, benchmark screenshot)
├── models/
│   └── blaze_face_short_range.tflite   # BlazeFace model (Apache-2.0, by Google)
├── outputs/            # Annotated images (generated; not committed)
├── review/             # Human-review queue (generated; not committed)
├── src/
│   ├── detect.py       # Two-stage detection + box drawing
│   ├── main.py         # Batch-process the dataset/ folder
│   └── evaluate.py     # 140k benchmark with CSV reports
├── requirements.txt
├── README.md
└── LICENSE
```

## Technologies

| | |
|---|---|
| [MediaPipe](https://developers.google.com/mediapipe) | BlazeFace short-range face detection (Tasks API) |
| [OpenCV](https://opencv.org/) | Image I/O, drawing, preprocessing |
| [NumPy](https://numpy.org/) | Image arrays |
| [kagglehub](https://github.com/Kaggle/kagglehub) | Benchmark dataset download |


## License

Released under the **MIT License** — see [LICENSE](LICENSE). The bundled `blaze_face_short_range.tflite` model is published by Google under the Apache License 2.0
