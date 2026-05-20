# Smart Factory Machinery Predictor Dataset Generator
# Creates a synthetic dataset with 2000 records for:
# 1. Linear Regression (predict Temperature)
# 2. Logistic Regression (predict Defective / Good part)

import pandas as pd
import numpy as np
import random

# Reproducibility
np.random.seed(42)
random.seed(42)

# Number of records
n = 2000

# -----------------------------
# Categorical Features
# -----------------------------
machine_modes = ["Eco", "Standard", "High-Speed"]
maintenance_levels = ["Poor", "Average", "Good"]
shift_types = ["Morning", "Evening", "Night"]

# -----------------------------
# Generate Input Features
# -----------------------------
data = {
    # Continuous Features
    "RPM": np.random.randint(800, 5000, n),                  # Rotations per minute
    "Vibration": np.round(np.random.uniform(0.1, 8.0, n), 2), # mm/s
    "Pressure": np.round(np.random.uniform(20, 120, n), 2),   # bar
    "Humidity": np.round(np.random.uniform(30, 90, n), 2),    # %
    "Voltage": np.round(np.random.uniform(210, 250, n), 2),   # Volts
    "Current": np.round(np.random.uniform(5, 50, n), 2),      # Amps
    "Load": np.round(np.random.uniform(10, 100, n), 2),       # %
    "Operating_Hours": np.random.randint(100, 10000, n),      # total machine hours

    # Categorical Features
    "Machine_Mode": np.random.choice(
        machine_modes, n, p=[0.25, 0.50, 0.25]
    ),
    "Maintenance_Level": np.random.choice(
        maintenance_levels, n, p=[0.20, 0.50, 0.30]
    ),
    "Shift_Type": np.random.choice(
        shift_types, n, p=[0.40, 0.35, 0.25]
    )
}

# Create DataFrame
df = pd.DataFrame(data)

# -----------------------------
# Encode Categorical Effects
# -----------------------------
mode_effect = {
    "Eco": -5,
    "Standard": 0,
    "High-Speed": 8
}

maintenance_effect = {
    "Poor": 6,
    "Average": 2,
    "Good": -3
}

shift_effect = {
    "Morning": 0,
    "Evening": 1,
    "Night": 3
}

# Random noise for realism
noise = np.random.normal(0, 2, n)

# -----------------------------
# Target 1: Temperature (Continuous)
# -----------------------------
df["Temperature"] = (
    25
    + 0.004 * df["RPM"]
    + 1.8 * df["Vibration"]
    + 0.12 * df["Pressure"]
    + 0.03 * df["Humidity"]
    + 0.06 * df["Current"]
    + 0.08 * df["Load"]
    + 0.0003 * df["Operating_Hours"]
    + df["Machine_Mode"].map(mode_effect)
    + df["Maintenance_Level"].map(maintenance_effect)
    + df["Shift_Type"].map(shift_effect)
    + noise
).round(2)

# -----------------------------
# Target 2: Defective (Binary Classification)
# 1 = Defective, 0 = Good
# -----------------------------
risk_score = (
    0.002 * df["RPM"]
    + 0.7 * df["Vibration"]
    + 0.05 * df["Pressure"]
    + 0.03 * df["Load"]
    + 0.0002 * df["Operating_Hours"]
    + df["Machine_Mode"].map({"Eco": -1, "Standard": 0, "High-Speed": 2})
    + df["Maintenance_Level"].map({"Poor": 3, "Average": 1, "Good": -2})
    + df["Shift_Type"].map({"Morning": 0, "Evening": 0.5, "Night": 1.5})
    + np.random.normal(0, 1.5, n)
)

# Convert risk score to probability using sigmoid
probability = 1 / (1 + np.exp(-(risk_score - 12)))

# Generate binary target
df["Defective"] = np.random.binomial(1, probability)

# -----------------------------
# Save Dataset
# -----------------------------
df.to_csv("smart_factory_machinery_data.csv", index=False)

# -----------------------------
# Display Information
# -----------------------------
print("Dataset created successfully!")
print("File saved as: smart_factory_machinery_data.csv")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDefective Class Distribution:")
print(df["Defective"].value_counts())

print("\nTemperature Statistics:")
print(df["Temperature"].describe())