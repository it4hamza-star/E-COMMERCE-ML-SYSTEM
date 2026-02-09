import joblib
import pandas as pd

def main():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = [f"feature_{i+1}" for i in range(20)]
    values = []

    print("Enter 20 numerical feature values for the customer session:")

    for name in feature_names:
        while True:
            try:
                val = float(input(f"{name}: "))
                values.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    sample = pd.DataFrame([values], columns=feature_names)
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    if prediction[0] == 1:
        print("\n✅ Customer likely to purchase")
    else:
        print("\n❌ Customer unlikely to purchase")

if __name__ == "__main__":
    main()
