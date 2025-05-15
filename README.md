# URL Classifier and Predictor

This project provides a system to classify URLs using machine learning techniques. It includes:

- A **feature extractor** to convert raw URLs into numerical features
- A **training script** supporting various classifiers
- A **prediction script** to make inferences using trained models

---

## 🧰 Environment Setup

This project uses **conda** for environment and dependency management. To create the environment:

```bash
conda env create -f environment.yml
conda activate url-classifier
```
---
## 📦 Dataset Format & Configuration
Required Format

Datasets must be in Parquet format and contain the following columns:

    url — the input URL string

    label — the target class as a string, such as:

        "benign"

        "malicious"

        "phishing"

        (or other categories)

🛑 Do not convert labels to integers. The system will handle encoding internally during training and prediction.
---
## Specifying a Dataset

You do not pass file paths directly via --dataset_name.
Instead, provide the name of an enum constant, which maps to the dataset path.

Example:
```python
import enum

class TrainDS(enum.Enum):
    grambeddings = "datasets/train.parquet"
```




## Training a model
```bash
python feature_train.py \
    --model_type random_forest \
    --dataset_name dataset_name \
    --seed 42
```
### Arguments
--model_type — Type of model to train:

    svm, random_forest, xgboost, logistic_regression, gradient_boosting, extra_trees

--dataset_name — Enum name of the dataset (e.g., grambeddings)

--seed — (Optional) Random seed for reproducibility (default: 42)
## Making Predictions
```bash
python feature_predict.py \
    --model models/xgb_model.pkl \
    --dataset_name grambeddings
```

## 📌 Notes

    Define dataset enums yourself to point to your actual .parquet files. Use different enums for train and test

    Ensure .parquet files contain:

        url column: raw URL strings

        label column: string-based class labels (e.g., "benign", "malicious")

    Feature extraction is part of the training and prediction pipeline.
    

