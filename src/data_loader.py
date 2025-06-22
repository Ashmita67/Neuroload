# src/data_loader.py

import os
import pandas as pd
import arff  # from liac-arff

# Define input/output paths
input_path = "data/raw/eeg_eye_state.arff"
output_path = "data/interim/eeg_eye_state.csv"

# Create interim folder if not exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load the .arff file
with open(input_path, 'r') as f:
    arff_data = arff.load(f)

# Convert to DataFrame
df = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

# Save as .csv
df.to_csv(output_path, index=False)

print(f"✅ EEG data converted to CSV and saved to {output_path}")
print(f"📐 Shape: {df.shape}")
