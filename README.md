#  Real Time Fraud Prevention — Credit Card Fraud Detection Web App

A clean, locally-hosted Flask web app that lets you simulate real-time credit card transaction screening using your trained Random Forest model.
<img width="1911" height="1011" alt="image" src="https://github.com/user-attachments/assets/e4355922-1026-4555-a49e-c479a340b38d" />
<img width="1912" height="1002" alt="image" src="https://github.com/user-attachments/assets/57112900-7258-48aa-ba37-f1354bf1b0aa" />
<img width="1914" height="901" alt="image" src="https://github.com/user-attachments/assets/6089840f-0b15-4909-a634-1207009cb092" />


---

##  Project Structure

```
fraud_app/
├── app.py                  ← Flask backend
├── model.pkl               ← Your trained model (copy here)
├── requirements.txt
└── templates/
    └── index.html          ← Frontend UI
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

>  Your `model.pkl` was trained with scikit-learn 1.2.2 — use **exactly** that version to avoid compatibility issues.

### 2. Place your model

Copy `model.pkl` into the `fraud_app/` folder (same directory as `app.py`).

### 3. Run the app

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

---

## 🎛️ How It Works

The model was trained on 30 features: `V1–V28` (PCA-reduced) + `Normalized Amount`. Since real users can't enter 28 PCA values, the app maps **4 human-readable inputs** to realistic feature patterns:

| Input | Description |
|---|---|
| **Amount ($)** | Transaction value, normalized to match training distribution |
| **Merchant Type** | Category (grocery, ATM, luxury, etc.) — maps to PCA patterns |
| **Location** | Local vs. International — foreign transactions use high-risk vectors |
| **Time of Day** | Daytime vs. Night-time — night transactions use riskier feature patterns |

These combinations represent real behavioral patterns observed in the original Kaggle dataset (e.g., ATM withdrawals at night are a strong fraud signal).

---

##  Model Info

- **Algorithm**: Random Forest Classifier (SMOTE-balanced)
- **Dataset**: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,315 transactions
- **Accuracy**: 99.98% | **Precision**: 99.97% | **Recall**: 1.0
