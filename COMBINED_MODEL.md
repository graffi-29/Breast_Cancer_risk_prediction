# Combined Mammogram + Ultrasound Demo

Start the combined website from the project root:

```powershell
python -m streamlit run combined_app.py --server.port 8504
```

Open `http://localhost:8504` and upload:

- One mammogram lesion crop from the DDSM, INbreast, or MIAS-style folders.
- One BUSI-style breast ultrasound image.

The page shows the mammogram malignant signal, ultrasound malignant signal, and an equal-weight average when both are present.

This is an ensemble of independently trained models. The datasets do not provide verified mammogram-ultrasound pairs from the same patients, so the combined score is not a calibrated medical-risk probability. It is a research/demo signal only.