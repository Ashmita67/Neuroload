import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Load the CSV
df = pd.read_csv("data/interim/eeg_eye_state.csv")

# Basic info
print("✅ Data loaded")
print(df.info())
print("\n📐 Shape:", df.shape)

# First few rows
print("\n🧾 First 5 rows:")
print(df.head())

# Column summary
print("\n📊 Descriptive Stats:")
print(df.describe())

# Class distribution
print("\n🔢 Class Distribution:")
print(df['eyeDetection'].value_counts())

# Plot class distribution
sns.countplot(x='eyeDetection', data=df)
plt.title("Target Distribution (Eye Open=0, Eye Closed=1)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("data/interim/class_distribution.png")
plt.show()

# Check for missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())

# Correlation heatmap (just EEG signals)
plt.figure(figsize=(12, 10))
sns.heatmap(df.iloc[:, :-1].corr(), cmap='coolwarm', annot=False)
plt.title("EEG Channel Correlations")
plt.tight_layout()
plt.savefig("data/interim/eeg_correlation.png")
plt.show()
