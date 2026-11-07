# ASIMOV

> 📌 **Project status update (November 2026)**  
> ASIMOV is being revisited after a period of inactivity. The initial focus is on reviewing the original implementation, updating its dependencies and documentation, and establishing a reliable baseline before further development.

> Experimental face detection and recognition using Deep Learning, with RetinaFace for detection and ArcFace for facial representation and comparison.

ASIMOV is a Python-based face detection and recognition project built around **TensorFlow 2**, **RetinaFace**, and **ArcFace**. The repository contains an experimental implementation of the detection and recognition components of a larger project.

The goal is to provide a straightforward interface for detecting faces in images, generating facial representations, and comparing faces, while keeping the underlying models accessible for experimentation.

The original implementation includes:

- Face detection and bounding-box annotation
- Face recognition using learned facial representations
- Face-to-face similarity comparison
- Image, video, and webcam testing utilities
- A graphical interface built on top of the API

---

## 🔍 Overview

ASIMOV separates **face detection** from **face recognition**.

**RetinaFace** locates faces in an input image. **ArcFace** then provides the recognition component, allowing face crops to be represented and compared.

The `API` class exposes three main operations:

| Method | Purpose |
| --- | --- |
| `detect_faces(frame, draw_bounding_box)` | Detect faces and optionally annotate the image |
| `recognize_faces(image)` | Compare a face crop against known faces |
| `compare_faces(first_crop, second_crop)` | Compare two face crops for similarity |

Detection and recognition can be enabled through the application's configuration. Model locations are defined separately in `Settings.py`.

The existing code and configuration reflect the project's original development environment. Compatibility with current Python, TensorFlow, CUDA, and operating-system versions has **not yet been revalidated**.

---

## 🚀 Quickstart

Clone the repository:

```bash
git clone https://github.com/JamesPiggott/ASIMOV.git
cd ASIMOV
```

Create and activate a virtual environment, then install the repository's dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Compatibility note:** These commands describe the standard setup process, not a verified installation on current systems. The original project used TensorFlow 2.x and had version-specific CUDA requirements. Check dependency compatibility before attempting GPU acceleration.

### API example

The following example illustrates the original API interface:

```python
import cv2
from application.core.API import API

image = cv2.imread("example.jpg")
api = API()

# Detect faces and request an annotated image.
result = api.detect_faces(image, True)

# Recognition expects an individual face crop.
# face_crop = ...
# match = api.recognize_faces(face_crop)
```

The detector returns a result containing face information, bounding boxes, and an annotated image. Recognition and comparison return match information, including a distance value and match indicator.

---

## 🧪 Testing

The original repository includes several test scripts:

| Script | Description |
| --- | --- |
| `video_detection_test.py` | RetinaFace detection on sample video |
| `webcam_test.py` | Live detection using a webcam |
| `image_recognition_test.py` | Face cropping, alignment, and ArcFace comparison |
| `GUI.py` | Interactive interface built on the API |

These scripts are useful starting points for examining the implementation. Their compatibility with a modern environment still needs to be checked.

---

## 🧭 Project Status

ASIMOV is a **revived experimental project**, currently at the assessment and modernization stage.

Original capabilities:

- ✅ RetinaFace detection integration
- ✅ ArcFace recognition integration
- ✅ Detection, recognition, and comparison API methods
- ✅ Example image/video/webcam testing utilities
- ✅ Experimental GUI

Planned work:

- 🔄 Audit and update Python dependencies
- 🔄 Validate model loading and inference
- 🔄 Revisit TensorFlow and GPU compatibility
- 🔄 Review configuration and model-path handling
- 🔄 Establish reproducible tests and benchmarks

The checkmarks indicate functionality represented in the original repository, **not functionality recently verified on current platforms**.

---

## 🎯 Goals

The immediate objective is to restore a maintainable, reproducible baseline. Potential subsequent work includes:

- Consistent face detection and alignment pipelines
- Reproducible face-embedding comparisons
- Clearer model configuration and loading
- CPU and GPU inference benchmarking
- Automated tests for the public API
- Evaluation of modern detection and recognition alternatives
- Documentation of accuracy, limitations, and operating requirements

These are development directions rather than promises of implemented features.

---

## ⚙️ Hardware and Dependencies

The original implementation was developed with Python 3, TensorFlow 2, and NVIDIA CUDA acceleration in mind.

GPU support depends on the combination of operating system, GPU drivers, TensorFlow version, CUDA toolkit, and cuDNN. Older installation instructions should not be assumed to work unchanged with modern releases.

Refer to the [official TensorFlow installation documentation](https://www.tensorflow.org/install/pip) when rebuilding the environment. A tested dependency matrix will be added after the implementation has been validated.

---

## 📚 Research and Acknowledgements

The original implementation builds on the work of the following projects:

- [RetinaFace for TensorFlow 2](https://github.com/peteryuX/retinaface-tf2)
- [ArcFace for TensorFlow 2](https://github.com/peteryuX/arcface-tf2)

Credit belongs to the original researchers and open-source maintainers whose work made these components possible. Model licenses, weights, and third-party dependencies should be reviewed before redistribution or deployment.

---

## 🤝 Contributing

ASIMOV is a research-oriented project. Contributions relating to model compatibility, reproducibility, testing, documentation, performance, and responsible evaluation are welcome.

---

## ⚠️ Disclaimer

ASIMOV is experimental software and is not presented as a production-ready biometric identification system.

Face recognition results can be affected by image quality, pose, illumination, demographic variation, model training data, and decision thresholds. A similarity score or match indicator should not be treated as definitive proof of identity.

Any use involving real people or biometric data must account for consent, privacy, security, and applicable legal requirements. Do not deploy this software for consequential identification or surveillance without appropriate validation and safeguards.
