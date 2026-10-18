# Student_Performace_Prediction
Feature Selection + SMOTE + Random Forest + XGBoost + Stacking Ensemble

An Advanced Machine Learning mini project (T.E. Sem VI, Vidyalankar Institute of Technology, Department of Computer Engineering) that classifies student academic performance into four categories: **Average, Good, Better, Best**. It uses a multi-stage pipeline: Information Gain and RFE feature selection, SMOTE class balancing, tuned Random Forest and XGBoost classifiers, and a Stacking Ensemble.

> Full project report: [`docs/Student_prediction_AML_report.docx`](docs/Student_prediction_AML_report.docx)

---

## Features

- Modular command-line pipeline, from raw CSV to saved model
- Automatic preprocessing for numerical and categorical columns
- Two feature-selection strategies: **Information Gain** (mutual information) and **Recursive Feature Elimination (RFE)**
- **SMOTE applied only to the training split**, so there is no data leakage
- Random Forest and XGBoost tuned with **GridSearchCV / RandomizedSearchCV** (3-fold stratified CV)
- **Stacking Ensemble**: RF + XGBoost, combined by a Logistic Regression meta-learner
- Evaluation with Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix and feature-importance plots
- Inference script for predicting on new student records

## Dataset

| | |
| --- | --- |
| Name | Students Performance in Exams |
| Source | UCI Machine Learning Repository / Kaggle |
| Records | 1,000 |
| Input features | gender, race/ethnicity, parental level of education, lunch, test preparation course |
| Target | Performance score binned into 4 classes (Average, Good, Better, Best) |
| Missing values | None |
| File | `data/student_performance.csv` |

## Tech Stack

| Category | Tools |
| --- | --- |
| Machine learning | scikit-learn, XGBoost |
| Feature selection | Mutual Information, RFE |
| Class balancing | SMOTE (imbalanced-learn) |
| Tuning | GridSearchCV, RandomizedSearchCV |
| Data processing | NumPy, pandas |
| Visualisation | Matplotlib, Seaborn |
| Model persistence | joblib (`.pkl`) |
| Language | Python 3.10+ |

## Pipeline

1. **Data loading and target engineering**: load CSV, bin the score into four quartile-based classes
2. **Preprocessing**: median imputation, one-hot encoding, 80/20 stratified train-test split
3. **Baseline model**: default Random Forest
4. **Information Gain** feature selection
5. **RFE** feature selection (Random Forest estimator)
6. **SMOTE** on the training split only
7. **Model training**: Random Forest and XGBoost
8. **Hyperparameter optimisation**: GridSearchCV (RF), RandomizedSearchCV (XGBoost)
9. **Stacking Ensemble** with a Logistic Regression meta-learner
10. **Evaluation and persistence**: all models scored on the held-out test set, best model by weighted F1 saved to `outputs/final_model.pkl`

## Project Structure

```
Student-Performance-Prediction/
├── train.py                    # Full training pipeline
├── predict.py                  # Inference using the saved model
├── data/
│   └── student_performance.csv # Source dataset (1,000 records)
├── outputs/                    # Saved model, charts, result tables
├── docs/                       # Project report
├── requirements.txt
└── README.md
```

## Setup (Windows / PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/chirayupritmani/<repo-name>.git
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
# If you get an execution-policy error:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

**Train all models and save the best one:**

```powershell
python train.py
```

This writes the trained model, confusion matrix, feature-importance chart, model-comparison chart and CSV result tables to `outputs/`.

**Predict on new records:**

```powershell
python predict.py
```

`predict.py` loads `outputs/final_model.pkl` and predicts the performance class for new student records.

## Results

Evaluated on the 20% held-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Baseline Random Forest | 35.0% | 37.1% | 35.0% | 35.5% | 0.571 |
| RF + Info Gain + SMOTE | 32.0% | 34.9% | 32.0% | 32.4% | 0.577 |
| XGBoost + Info Gain + SMOTE | 33.0% | 33.9% | 33.0% | 32.2% | 0.581 |
| Optimised Random Forest | 33.5% | 34.7% | 33.5% | 32.6% | 0.581 |
| Optimised XGBoost | 33.5% | 34.7% | 33.5% | 32.6% | 0.584 |
| **Stacking Ensemble** | **35.5%** | 35.8% | **35.5%** | 34.5% | **0.584** |

**Key observations**

- The Stacking Ensemble has the highest accuracy and ROC-AUC (0.584).
- The Baseline Random Forest has the highest weighted F1 (35.5%), so SMOTE and feature selection do not uniformly help on this small dataset.
- `race/ethnicity` and `parental level of education` are the most influential predictors.
- Accuracy is modest because this is a noisy four-class problem with only 1,000 records. An AUC above 0.5 (random) shows the models still learn real class structure.

## Limitations and Future Work

- Small dataset and a limited feature set restrict generalisation
- Richer features (attendance, assignment scores) would likely help
- Try deep tabular models such as TabNet
- Deploy as a web-based decision-support tool for educators

## Team

| Roll No. | Name |
| --- | --- |
| 23102C0085 | Chirayu Pritmani |
| 23102C0084 | Samruddhi Gawade |
| 23104B0023 | Swar Sawalkar |

**Subject:** Advanced Machine Learning | **Guide:** Prof. Sneha Annappanavar | **Institute:** Vidyalankar Institute of Technology, Mumbai

## License

For academic purposes.
