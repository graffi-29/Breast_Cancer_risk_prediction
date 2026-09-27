# Breast Cancer Classification

An educational machine-learning project using scikit-learn's Wisconsin Diagnostic Breast Cancer dataset. It compares logistic regression, an RBF-kernel support vector machine, and random forest, then evaluates the candidate with the highest mean cross-validation ROC-AUC on a held-out test split.

The data has 569 samples and 30 numeric measurements. The project maps malignant tumors to the positive class (`1`) so malignant recall is reported directly. Scaling is part of each model pipeline, so it is fitted within each cross-validation fold and cannot leak validation or test data into training.

## Setup

Use Python 3.10 or newer. From this folder, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dataset is bundled with scikit-learn, so no separate dataset download is needed.

## Train and evaluate

```powershell
python train.py
```

Optional settings:

```powershell
python train.py --test-size 0.2 --random-state 42 --cv-folds 5 --output-dir artifacts
```

The script prints each candidate's cross-validation ROC-AUC, the selected model's test metrics, malignant recall, benign recall (specificity), and a confusion matrix. It saves the fitted pipeline to `artifacts/best_model.joblib` and detailed metrics to `artifacts/metrics.json`.

## Open the presentation dashboard

Train the model first, then start the dashboard:

```powershell
python -m streamlit run app.py
```

The dashboard compares cross-validation ROC-AUC, shows the held-out test ROC curve and confusion matrix, and highlights sensitivity and specificity. It uses the saved model and metrics from `artifacts/`.

## Run the test

```powershell
python -m unittest discover -v
```

## Reading the results

- **Malignant recall/sensitivity** is the fraction of malignant cases the model identifies. Missed malignant cases are false negatives.
- **Benign recall/specificity** is the fraction of benign cases correctly identified.
- **ROC-AUC** summarizes ranking performance across classification thresholds; it does not determine a clinically appropriate threshold.
- The confusion matrix rows are actual labels and columns are predicted labels, ordered benign then malignant.

Model selection uses cross-validation on the training split. The test split is held out until selection is complete. Results can vary with the split and do not establish performance on other populations or clinical settings.

## Important limitation

This is a learning exercise, not a medical device or diagnostic tool. The dataset is small and does not represent every patient population or care setting. Do not use the model for medical decisions.

## Ultrasound image model

The image version uses only the BUSI ultrasound portion of MammoBench v2. It is separate from the numeric Wisconsin model above. Download the Kaggle dataset from [MammoBench v2](https://www.kaggle.com/datasets/dhruvkarani22/mammobench-v2), then place the ultrasound class folders here:

```text
data/ultrasound/images/benign/
data/ultrasound/images/malignant/
data/ultrasound/images/normal/
```

The archive may use a different top-level folder name. Copy or move the three class folders into the paths above. The image trainer accepts JPG, JPEG, and PNG files.

Install the additional image dependencies and train:

```powershell
python -m pip install -r requirements.txt
python train_ultrasound.py --data-dir data/ultrasound/images
```

Then launch the ultrasound upload app:

```powershell
python -m streamlit run ultrasound_app.py
```

Upload a breast ultrasound image similar to the BUSI training images. The app returns one of `benign`, `malignant`, or `normal` with the model confidence. It cannot analyze mammograms, photographs, scans from other modalities, or arbitrary medical images. The result is an educational demonstration and is not a diagnosis.