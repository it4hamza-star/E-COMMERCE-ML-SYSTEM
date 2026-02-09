import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(X):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Keep as DataFrame with column names to remove warnings
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, scaler
