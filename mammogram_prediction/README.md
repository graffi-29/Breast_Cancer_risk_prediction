# Mammogram Prediction Model

This is a separate mammogram-only project using the individual DDSM, INbreast, and MIAS folders from the supplied dataset. It excludes `INbreast+MIAS+DDSM Dataset` because that folder appears to contain a combined copy and could duplicate images.

## Train

From the main project folder:

```powershell
python mammogram_prediction/train_mammogram.py --dataset-root "C:/Users/graff/Downloads/Dataset of Mammography with Benign Malignant Breast Masses"
```

The trainer uses MobileNetV2 transfer learning and grouped filename-based splitting. It saves `artifacts_mammogram/mammogram_model.keras` and `artifacts_mammogram/mammogram_metrics.json`.

## Open the upload website

```powershell
python -m streamlit run mammogram_prediction/mammogram_app.py --server.port 8503
```

Upload a PNG, JPG, or JPEG mammogram lesion crop similar to the supplied dataset. The result is a benign/malignant model signal. It is not a calibrated individual cancer-risk percentage or a diagnosis.

## Important limitations

The dataset is assembled from multiple sources, images may be repeated or related, and filename grouping is only a practical leakage-reduction measure. External patient-level validation is still required. A model score is not clinical risk and must not be used for medical decisions.