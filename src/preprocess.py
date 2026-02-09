import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, scaler

if __name__ == "__main__":
    import dataset
    X, _ = dataset.generate_dataset()
    X_scaled, _ = preprocess(X)
    print(X_scaled.head())
