# Skin Disease Detection — Project Documentation

## Overview
This project is a complete end-to-end prototype for skin disease detection using image classification and retrieval. It includes training/inference code, a Flask-based web UI, a dataset embedding pipeline for fast nearest-neighbor matching, and a simple chat assistant augmented with retrieval.

Key goals:
- Classify uploaded skin images using a fine-tuned MobileNet model.
- Quickly find the most similar images from the dataset and show them to users.
- Provide actionable suggestions, medicines, and home remedies per class.
- Offer a chat assistant that can answer follow-ups and show dataset examples.

## Repository layout
- `app.py` — Flask web server with endpoints for prediction, matched image serving, and chat.
- `build_embeddings.py` — Script that computes MobileNetV2 embeddings for dataset images and writes `model/embeddings.npy` and `model/embeddings_meta.json`.
- `model/` — Model and artifacts directory (contains `skin_disease_model.h5`, `embeddings.npy`, `embeddings_meta.json`, `suggestions.json`, and `class_indices.json`).
- `templates/` — HTML templates (`login.html`, `dashboard.html`).
- `static/` — Static frontend assets (CSS, JS, images, icons).
- `static/js/app.js` — Client-side logic for upload, prediction, chat, and rendering retrieved examples.
- `static/css/style.css` — Styling for the dashboard and chat UI.
- `test_upload.py` — Lightweight script to POST a file to `/predict` (used for local tests).

## Technologies used
- Python 3.x, Flask for the web server and API.
- TensorFlow / Keras for model training and inference.
- Pillow for image loading/manipulation.
- NumPy for numeric operations and embeddings.
- scikit-learn (optional) for TF-IDF retrieval over dataset text.
- Bootstrap + vanilla JS for frontend UI.

## Data and dataset structure
- Expected dataset root: `model/` with `train/` and `test/` subfolders.
- `build_embeddings.py` walks through `model/train` and `model/test`, computes embeddings for each image, and writes `model/embeddings.npy` (float32 array of embedding vectors) and `model/embeddings_meta.json` (list of metadata: path, label).
- `suggestions.json` maps class labels to human-friendly suggestions, medicines, care types, and home remedies used by the chat assistant and dashboard.

## Model architecture and prediction pipeline
- Backbone: MobileNetV2 from Keras applications. The project uses two roles for MobileNetV2:
  - As a classification model (`skin_disease_model.h5`) fine-tuned on the dataset (top layers trained or retrained).
  - As an embedding extractor (`include_top=False, pooling='avg'`) to compute dataset and query image vectors for nearest neighbor search.

- Transfer learning strategy:
  - The pretrained ImageNet MobileNetV2 is used as a feature extractor; a small classification head is attached and trained on the dataset.
  - Training uses standard data augmentation (random flips, rotations, brightness jitter) where implemented in training scripts.
  - Multi-GPU support (if present) via `tf.distribute.MirroredStrategy` and optional mixed precision for faster training.

- Prediction (runtime):
  - Uploaded image is resized to 224x224, converted to RGB, preprocessed with `mobilenet_v2.preprocess_input`, then fed to the classification model.
  - The model outputs a probability vector over classes; the server returns the top-3 predictions (class, confidence) as JSON.

Algorithms and methods used for prediction quality and reasoning
- Softmax probabilities from the trained classifier are used as confidence estimates.
- To provide context when the top probability is low, the API returns the top-K (default 3) classes and probabilities.
- Optional calibration: temperature scaling or Platt scaling may be applied to recalibrate probabilities using a held-out validation set (not enabled by default). See "Calibration" below.

## Embeddings and nearest-neighbor matching
- Embedding extractor: MobileNetV2 (`include_top=False`, `pooling='avg'`) produces fixed-length vectors per image.
- Embeddings stored in `model/embeddings.npy` are L2-normalized so cosine similarity reduces to dot product.
- Nearest neighbor search: query embedding (L2-normalized) is dot-producted with all dataset embeddings to compute similarity scores; the highest scoring item is returned as the most similar dataset example.
- Performance: linear scan is fast enough for datasets with tens of thousands of images in CPU/desktop environments; for production-scale datasets, approximate nearest neighbor libraries (FAISS, Annoy, hnswlib) are recommended.

## Retrieval-augmented chat assistant
- The `/chat` endpoint implements a lightweight rule-based assistant that uses `model/suggestions.json` for label-aware answers (home remedies, medicines, key care tips, escalation advice).
- Retrieval augmentation:
  - A TF-IDF vectorizer (scikit-learn) builds a document matrix over dataset metadata (label, filename, parent folder).
  - When scikit-learn is available and the index is built, the chat handler transforms the user question (optionally prefixed by the predicted label) and computes cosine similarity against the TF-IDF matrix to find top dataset text matches.
  - The API returns these retrieved items with `path`, `label`, and `score` to the frontend; the UI shows thumbnails and clickable links to the matched images.

## Flask endpoints
- `/login` — GET/POST login page (demo credentials). For production, replace with a real auth system.
- `/dashboard` — main UI, requires login.
- `/predict` — POST with multipart form `file`. Returns JSON: `class`, `confidence`, `top` (top-K list), and `matched_path`/`matched_label`/`similarity` if embeddings are available, plus `suggestions` data.
- `/matched` — GET serving dataset image files by relative path (used by the UI to display thumbnails).
- `/chat` — POST with JSON `{question, label(optional)}`; returns rule-based answer and optional `retrieved` list when TF-IDF is available.

## Frontend behavior
- Dashboard allows selecting an image file; the preview updates via `URL.createObjectURL`.
- `Predict` uploads the file to `/predict` and displays top-3 predictions and the most similar dataset image.
- Chat UI submits follow-up questions to `/chat`. If dataset retrieval is enabled, the chat shows thumbnail grid of retrieved examples; caption links open the example in a new tab.

## How to run (development)
1. Create and activate your Python environment and install dependencies (suggested):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure the model and artifacts exist under `model/`: `skin_disease_model.h5`, `class_indices.json`, and (optionally) `embeddings.npy` and `embeddings_meta.json`.
3. Build embeddings (if not present):

```powershell
python build_embeddings.py
```

4. Start the server:

```powershell
python app.py
```

5. Open `http://127.0.0.1:5000` in a browser, login with demo credentials and use the dashboard.

## Calibration (optional)
- Temperature scaling is a post-processing step to calibrate softmax probabilities.
- Steps to enable:
  1. Hold out a validation dataset with ground truth labels.
  2. Run the model to get logits and labels on the validation set, and fit a single scalar temperature `T` that minimizes negative log-likelihood on the validation logits after scaling (`softmax(logits / T)`).
  3. Save `T` and apply at inference: use `softmax(logits / T)` instead of `softmax(logits)`.

I can add a small utility to fit and persist temperature if you want — it requires labelled validation data.

## Improving accuracy and confidence
- Data: more labeled examples, better class balance, and higher-quality labels improve model certainty.
- Architecture: try stronger backbones (EfficientNetV2, ResNet variants) or ensemble multiple models.
- Augmentation: advanced augmentations and domain-specific transforms can help generalization.
- Calibration: temperature scaling (as above) or isotonic regression on validation data.
- Ensembles: average probabilities from multiple independently trained models to improve confidence.

## Security, privacy, and disclaimers
- The service is a prototype — not for clinical use. All medical suggestions are informational only.
- Do not store or transmit PII without proper consent and secure storage.
- For production, add HTTPS, proper authentication, rate limiting, and input validation.

## Limitations
- Model mistakes: misclassification is possible; probabilities may be overconfident without calibration.
- Dataset bias: results depend on training data distribution (skin tones, imaging conditions).
- Resource usage: TensorFlow model loading and embeddings can be memory/CPU intensive.

## Next steps and roadmap
- Add temperature scaling calibration using a labelled validation set.
- Replace TF-IDF with embedding-based ANN (FAISS/hnswlib) for semantic retrieval at scale.
- Add user accounts, upload history, and secure image storage.
- Integrate an LLM for richer conversational answers (with guardrails and retrieval augmentation).
- Add unit tests and CI for model check, endpoint responses, and UI smoke tests.

## References
- MobileNetV2 paper: https://arxiv.org/abs/1801.04381
- TensorFlow Keras Applications: https://www.tensorflow.org/api_docs/python/tf/keras/applications
- FAISS: https://github.com/facebookresearch/faiss
- scikit-learn TF-IDF: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

---

File references:
- `app.py` — server and endpoints
- `build_embeddings.py` — embedding pipeline
- `model/suggestions.json` — suggestions used by the chat and dashboard
- `model/embeddings.npy`, `model/embeddings_meta.json` — precomputed dataset embeddings
- `templates/dashboard.html`, `static/js/app.js`, `static/css/style.css` — frontend UI

If you want, I can also generate a polished `README.md` with quick-start commands and add a `docs/` folder with separate pages for Training, Inference, and Retrieval internals.
