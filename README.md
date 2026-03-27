# 🧬 CodeCureAI — Tox21 Molecular Toxicity Predictor

🚀 An end-to-end AI system for predicting **molecular toxicity** across 12 biological endpoints using an advanced **stacking ensemble model** with explainability.

---

## 🌐 Live Demo

👉 **Streamlit App:**
https://codecureai26-htcymhbvj4f8tzvdwwhluc.streamlit.app/

---

## 📌 Project Overview

**CodeCureAI** is designed to:

* Predict toxicity of chemical compounds from **SMILES strings**
* Provide **probability-based risk analysis**
* Deliver **interpretable insights using SHAP**
* Support both:

  * 🧪 Research workflow (Notebook)
  * 🌐 Web deployment (Streamlit + HTML UI)

---

## 🧠 Model Architecture

We use a **Stacking Ensemble Model**:

```
Base Models:
  ├── Random Forest
  ├── XGBoost
  └── LightGBM

Meta Model:
  └── Logistic Regression
```

### 🔥 Key Improvement

* Replaced **SVM** with **LightGBM**
* Achieved better performance with:

  * ⚡ Faster training
  * 📉 Smaller model size
  * 📈 Higher ROC-AUC

---

## 📊 Results

| Metric           | Score              |
| ---------------- | ------------------ |
| **Mean ROC-AUC** | **0.8613 (~0.86)** |
| Mean PR-AUC      | 0.5365             |
| Mean Accuracy    | 0.8871             |

### 🏆 Highlights

* **Best Target:** `SR-MMP` → ROC = **0.9279**
* **Worst Target:** `NR-AR` → ROC = **0.7652**
* Improved from **0.8528 → 0.8613**

---

## ⚙️ Feature Engineering

Each molecule is converted into **4719 features** using:

* Morgan Fingerprints (ECFP4)
* MACCS Keys
* RDKit Fingerprints
* Topological Torsion
* 200 Molecular Descriptors

---

## 🔬 Training Pipeline

* Dataset: **Tox21**
* Preprocessing:

  * Variance Thresholding
  * Robust Scaling
* Handling imbalance:

  * `scale_pos_weight` per target
* Validation:

  * 5-Fold Stratified Cross Validation
* Stacking:

  * Out-of-Fold (OOF) predictions

📁 Full training workflow available in:

```
codecure.ipynb
```

---

## 🖥️ Applications

### 1️⃣ Streamlit App (AI UI)

* File: `streamlitapp.py`
* Features:

  * SMILES input
  * Toxicity prediction
  * SHAP explainability
  * Interactive UI

---

### 2️⃣ HTML Web App

* File: `app.py`
* Built using:

  * HTML
  * CSS
  * JavaScript

---

## 🔄 System Flow

```
User Input (SMILES)
        ↓
Feature Engineering (RDKit)
        ↓
Stacking Model (RF + XGB + LGBM)
        ↓
Meta Learner (Logistic Regression)
        ↓
Prediction Output (12 Targets)
        ↓
SHAP Explainability
        ↓
Visualization (Streamlit UI)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment (Python 3.11)

```
python3.11 -m venv venv
```

### 2️⃣ Activate Environment

**Windows:**

```
venv\Scripts\activate
```

**Linux / Mac:**

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 🔹 Run Training Pipeline

```
python main.py
```

* Executes full ML pipeline
* Trains model
* Saves outputs in:

```
artifacts/
```

---

### 🔹 Run HTML Web App

```
python app.py
```

* Launches basic web interface (HTML/CSS/JS)

---

### 🔹 Run Streamlit UI

```
streamlit run streamlitapp.py
```

* Launches interactive AI dashboard
* Includes:

  * Predictions
  * Visualizations
  * SHAP Explainability

---

## 🚀 Deployment

* Streamlit Cloud used for hosting UI
* Model loaded dynamically from Hugging Face
* Supports real-time predictions

---

## 🧪 Explainability (SHAP)

* Feature importance visualization
* Waterfall plots
* Per-molecule interpretability

---

## 🛠️ Tech Stack

* Python
* Streamlit
* RDKit
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Hugging Face

---

## 🔥 Key Contributions

✔ Built high-performance toxicity prediction model
✔ Improved ROC-AUC using LightGBM stacking
✔ Integrated explainability (SHAP)
✔ Deployed full-stack AI system
✔ Built dual UI (Streamlit + HTML)

---

## 📌 Future Improvements

* Convert model to API (FastAPI)
* Add batch predictions
* Improve inference speed
* Add more chemical datasets

---

## 👨‍💻 Author

**Gowtham D**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
