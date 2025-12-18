# AI-Powered Meat Quality Analysis (Edge CV)

**Real-time meat freshness and marbling analysis on Raspberry Pi using TensorFlow Lite and classical computer vision.**

---

## Overview

Production-ready **edge vision pipeline** performing **dual analysis** on meat samples:

- **Freshness classification** via custom **TensorFlow Lite CNN**  
- **Marbling assessment** via **explainable computer vision algorithms**

Low-latency, interpretable, and reliable on ARM64 hardware.

---

## Key Features

- Edge-optimized **TFLite inference** (sub-second on Raspberry Pi 4)  
- **Hybrid ML + classical CV** architecture  
- Multi-color-space marbling analysis (RGB, HSV, LAB)  
- USDA-style grading (Prime / Choice / Select / Standard)  
- JSON outputs with confidence scores  
- Robust camera integration and result persistence  

---

## Technical Stack

| Layer              | Technology / Tool                  |
|-------------------|----------------------------------|
| **Hardware**       | Raspberry Pi 4 (4–8GB RAM), Pi Camera v2/HQ |
| **OS & Runtime**   | Raspberry Pi OS (Bookworm), Python 3.9+   |
| **ML Framework**   | TensorFlow Lite                        |
| **Computer Vision**| OpenCV, scikit-image, SciPy           |
| **Web & API**      | Flask, Socket.IO                      |
| **Service Mgmt**   | systemd (auto-restart, monitoring)    |
| **Camera API**     | Picamera2                             |
| **Data Format**    | JSON for results, JPEG for images     |

---

## Core Module

### `integrated_meat_analyzer.py`

Handles:
- Image capture (Pi Camera)
- Freshness inference (TFLite)
- Marbling segmentation and grading
- Result aggregation and serialization

---

## Technical Approach

### Freshness (ML)
- Custom CNN → TensorFlow Lite
- Classes: `FRESH`, `HALF`, `SPOILED`
- Optimized for ARM via TFLite runtime

### Marbling (Computer Vision)
- Meat segmentation: HSV + LAB
- Fat detection: brightness, saturation, RGB balance
- Marbling streak detection via connected components
- Rule-based grading (interpretable)

---

## Example Output

```json
{
  "freshness": { "class": "FRESH", "confidence": 0.94 },
  "marbling": { "grade": "Choice", "fat_percentage": 12.5 }
}

Author

Eden Yoseph — CV / ML Engineering | Edge AI & Computer Vision

License
Research and educational use.
