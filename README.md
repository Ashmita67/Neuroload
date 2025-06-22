# 🧠 NeuroLoad – EEG-Based Cognitive Load Detection

**NeuroLoad** is a machine learning system that detects cognitive load from multi-channel EEG signals.  
It uses spectral and statistical features from EEG windows to classify mental fatigue, helping in health monitoring, human–computer interaction, and workload assessment.

> 🔍 Achieved **85% accuracy** using a Random Forest classifier on the UCI EEG Eye State dataset.  
> ✅ Includes a Streamlit demo app for real-time load detection from uploaded EEG CSV files.

---

## 📌 Features

- 🧠 Cognitive load classification from 14-channel EEG data
- 🔬 Feature extraction using band power (alpha, beta, theta, delta) and statistical metrics
- 🎯 Trained using Random Forest (scikit-learn)
- 📊 Model explainability using SHAP plots
- 🌐 Interactive Streamlit web app for prediction and visualization
- 🗂 Clean modular structure for easy reproducibility

---

## 🚀 Demo

### ▶ Run the Web App:
```bash
streamlit run app/app.py

---
### Dataset Used
Name: EEG Eye State
Source: UCI Machine Learning Repository
Details:
1) 14 EEG channels
2) 14980 samples @ 128 Hz
3) Binary label: eye open (0) or closed (1), used as a proxy for load

---
## Methodology
Preprocessing: Bandpass filtering (1–45 Hz), windowing (2s with 50% overlap)
Feature Engineering: Bandpower (theta, alpha, beta), mean, variance, skew, kurtosis
Modeling: Random Forest classifier with 400 trees
Evaluation: 85% test accuracy, confusion matrix, classification report
Explainability: SHAP summary plots highlight important EEG bands/channels

---
### 🧑‍💻 How to Run
# 1. Clone the repo
git clone https://github.com/yourhandle/NeuroLoad.git
cd NeuroLoad
# 2. Setup virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
# 3. Install dependencies
pip install -r requirements.txt
# 4. Run the app
streamlit run app/app.py
---

## Future Work
Incorporate more cognitive tasks (e.g., memory, reaction time)
Apply on multi-subject datasets (e.g., SEED, DEAP)
Real-time EEG acquisition (e.g., Emotiv, Muse)
Add LSTM/GRU-based deep models for temporal learning

---
## 🌟 Acknowledgements
UCI Machine Learning Repository for the EEG dataset
Streamlit & SHAP community
Python libraries that made this possible