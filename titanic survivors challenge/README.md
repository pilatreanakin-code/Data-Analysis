# Titanic Survival Prediction — Kaggle Competition

End-to-end machine learning pipeline for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic). Predicts which passengers survived the Titanic disaster using passenger data (age, sex, class, etc.).

---

## Results

| Model | CV Accuracy (5-fold) |
|---|---|
| Logistic Regression (baseline) | ~80% |
| **Random Forest (submission)** | **~83%** |

Submission file: `output/submission.csv` — 418 predictions, ready to upload.

---

## Project Structure

```
titanic-kaggle/
├── data/                        # Drop train.csv and test.csv here (not tracked)
├── notebooks/
│   └── titanic.ipynb            # Full pipeline: EDA → features → models → submission
├── output/
│   ├── submission.csv           # Kaggle submission file
│   ├── survival_rates.png       # EDA: survival by sex, class, age, embarkation
│   ├── correlation_heatmap.png  # EDA: feature correlation matrix
│   ├── distributions.png        # EDA: age and fare distributions by survival
│   └── feature_importances.png  # Random Forest top-15 feature importances
├── models/
│   └── random_forest.pkl        # Trained Random Forest (200 trees, max_depth=6)
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

### 1. Feature Engineering
- **Title** extracted from passenger name (Mr, Mrs, Miss, Master, Rare)
- **FamilySize** = SibSp + Parch + 1; **IsAlone** flag
- **Deck** decoded from Cabin first letter (unknown → 'U')
- **AgeBin**: Child / Teen / Adult / Senior
- **FareBin**: quartile buckets Q1–Q4

### 2. Preprocessing
- Missing `Age` filled with median per title group
- Missing `Embarked` filled with mode
- Missing `Fare` (1 test row) filled with median
- One-hot encoding applied jointly on train + test to guarantee identical columns

### 3. Models
- **Baseline**: `statsmodels.Logit` — full summary table with p-values and confidence intervals
- **Main**: `sklearn.RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)`

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/titanic-kaggle.git
cd titanic-kaggle

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add data files (download from Kaggle)
#    Place train.csv and test.csv in data/

# 4. Execute the notebook
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=120 notebooks/titanic.ipynb

# 5. Find your submission at output/submission.csv
```

---

## Key Findings

- **Sex** is the strongest predictor — female passengers survived at ~74% vs ~19% for males
- **Pclass** is the second strongest — 1st class survival rate (~63%) nearly doubles 3rd class (~24%)
- **Title** adds signal beyond raw sex/age — `Master` (young boys) had high survival; `Mr` had the lowest
- **FamilySize** shows a non-linear pattern — solo travellers and very large families fared worse than medium-sized groups

---

## Dependencies

```
pandas · numpy · matplotlib · seaborn · scikit-learn · statsmodels · notebook · ipykernel
```

Python 3.10+, all pinned in `requirements.txt`.
