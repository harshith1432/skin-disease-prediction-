# Skin Disease Detection Using Image Classification

## Abstract
This project builds an image-classification system to detect common skin conditions (acne, rash, eczema, allergy, fungal infection) using transfer learning with MobileNet architectures. The system includes dataset structure, preprocessing, augmentation, training scripts, a Flask backend for inference, a modern web UI, and full MCA-level documentation.

## Introduction
Skin disease detection using AI can assist clinicians and users by offering rapid screening of common conditions. This project demonstrates building such a pipeline with explainable steps.

## Problem Statement
Manual diagnosis can be slow, subjective, and inaccessible in resource-limited settings. An automated classifier can provide triage and support.

## Objective
- Build a transfer-learning classifier for five skin disease categories.
- Provide a Flask API and a clean UI for image uploads and predictions.
- Supply training code, preprocessing, augmentation, and documentation for reproducibility.

## Existing System
Many solutions focus on melanoma detection (HAM10000). Few provide a lightweight web-deployable pipeline for common non-melanoma conditions.

## Proposed System
A web-based Flask app with a MobileNet-based CNN backend trained on a curated dataset, supporting upload, inference, and result display.

## Architecture Diagram
(Place diagram here in final report) - a simple architecture: User -> Browser UI -> Flask server -> Model -> Prediction

## Use Case Diagram
(Describe actors: User, Admin, Model Trainer)

## Data Flow Diagram
(High-level: Upload image -> Preprocess -> Predict -> Return results)

## ER Diagram
(Not applicable; minimal data persisted)

## Methodology
1. Collect and prepare dataset (folder-wise).
2. Preprocess images (resize to 224x224, normalize using MobileNet preprocessing).
3. Augment training images to reduce overfitting.
4. Train MobileNetV2 or MobileNetV3 with fine-tuning.
5. Evaluate and save best model.
6. Deploy model via Flask and provide UI.

## Model Training Details
- Input size: 224x224x3
- Backbone: MobileNetV2 (or MobileNetV3Small)
- Loss: Categorical Crossentropy
- Optimizer: Adam (1e-4)
- Checkpoint: save best by validation accuracy
- Typical epochs: 10-30 depending on dataset size

## Algorithms Used
- Convolutional Neural Network (transfer-learning)
- MobileNet family: depthwise separable convolutions for efficiency

## Results
- Place evaluation metrics after training (accuracy, precision, recall, F1, confusion matrix)

## Limitations
- Quality/size of dataset affects performance.
- Class imbalance will bias predictions.
- Model not a medical diagnosis; use for triage only.

## Future Enhancements
- Add Grad-CAM explanations.
- Expand classes and use larger datasets.
- Provide mobile app frontend.

## Conclusion
This project provides a full pipeline to build, train, and deploy a skin disease classifier using transfer learning. It includes code to reproduce results and a user-friendly UI for testing.

---

# Installation & Running

1. Create a Python virtual environment and activate it.

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepare dataset according to `dataset/README.md`.
3. Train model (example):

```powershell
python train.py --data_dir dataset --backbone mobilenetv2 --epochs 15 --batch 32 --save_to model/skin_disease_model.h5
```

4. Run Flask app:

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser.

# Deployment Notes
- For Render: create a `start` command in `render.yaml` or deploy a Flask web service, set `gunicorn app:app` as the start command and ensure `requirements.txt` is present.
- For PythonAnywhere: upload files, install requirements in a virtualenv, and point a web app to `app.py`.
- For Streamlit: this project uses Flask; to convert create a `app_streamlit.py` that loads the model and provides `st.file_uploader` UI.

