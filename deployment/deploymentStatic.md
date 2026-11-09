# Deployment Setup

## Overview

This project deploys a trained PyTorch CNN for plant disease classification using **Hugging Face Static Spaces**.

Since Static Spaces do not support running Python servers, the trained CNN is exported to ONNX and executed directly in the browser using **ONNX Runtime Web**.

The original Gradio interface remains available in `app/app.py` for local Python-based testing.

## 1. Project Structure

```text
deployment/
├── index.html          # Web interface
├── app.js              # Image preprocessing and CNN inference
├── model.onnx          # Exported trained CNN
├── labels.json         # Disease class names
├── model_config.json   # Model preprocessing configuration
└── SETUP.md            # Deployment instructions
```

## 2. Local Setup

**Requirements:**
- Python 3
- A modern web browser
- Internet connection to load ONNX Runtime Web

From the repository root, run:

```bash
python -m http.server 8000 --directory deployment
```

Open:

http://127.0.0.1:8000/

Upload a leaf image and select **Predict Disease**.

The application should display the five most likely disease classifications and their probabilities.

To stop the local server, press `Ctrl + C`.

## 3. Hugging Face Deployment

1. Create a new Hugging Face Space.
2. Select **Static** as the SDK.
3. Upload the contents of `deployment/` to the Space repository root.
4. Configure the Space's `README.md` with:

```yaml
---
title: Plant Disease Classification
emoji: 🌿
sdk: static
app_file: index.html
---
```

5. Commit the files and open the Space's **App** tab.
6. Test the classifier by uploading a leaf image.

No Python backend is required for browser inference.

## 4. Updating the CNN Model

When a new model checkpoint is available:

1. Export the trained PyTorch model to ONNX using `scripts/export_onnx.py`.
2. Replace `deployment/model.onnx` with the updated model.
3. Update `labels.json` and `model_config.json` if the model configuration has changed.
4. Verify that ONNX predictions match the original PyTorch model within an acceptable numerical tolerance.
5. Test locally before uploading the updated files to Hugging Face.

**Important:** Image preprocessing must match the preprocessing used during model training.

## 5. Troubleshooting

| Issue | Possible Solution |
|---|---|
| Model fails to load | Check that `model.onnx` exists and the browser console for errors. |
| Predictions are incorrect | Verify preprocessing, normalisation and class label ordering. |
| Blank webpage | Check `index.html`, `app.js` and the browser console. |
| Files cannot be fetched | Run a local HTTP server instead of opening `index.html` directly. |
| Hugging Face deployment fails | Verify the Static SDK configuration and file paths. |

## 6. Testing Checklist

- [ ] ONNX model loads successfully.
- [ ] Image upload and preview work.
- [ ] CNN produces real predictions.
- [ ] Top five predictions display correctly.
- [ ] Results are consistent with the original PyTorch model.
- [ ] Application works on Hugging Face.

## Notes

- The deployed application runs inference in the user's browser.
- Model weights are downloaded when the application loads.
- Prediction performance depends on the user's device.
- The original PyTorch and Gradio implementations are preserved in the main repository.
- Predictions are experimental and should not be treated as definitive plant disease diagnoses.