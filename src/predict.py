import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature names
feature_names = [f"feature_{i+1}" for i in range(20)]
values = []

print("Enter 20 feature values for the customer session:")

for name in feature_names:
    while True:
        try:
            val = float(input(f"{name}: "))
            values.append(val)
            break
        except ValueError:
            print("Please enter a valid number.")

# Convert to DataFrame with column names
sample = pd.DataFrame([values], columns=feature_names)

# Scale and predict
sample_scaled = scaler.transform(sample)
sample_scaled = pd.DataFrame(sample_scaled, columns=feature_names)

prediction = model.predict(sample_scaled)

if prediction[0] == 1:
    print("Customer likely to purchase")
else:
    print("Customer unlikely to purchase")
