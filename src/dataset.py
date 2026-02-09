import pandas as pd
from sklearn.datasets import make_classification

def generate_dataset(n_samples=1000, n_features=20, random_state=42):
    X_array, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=random_state
    )
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    X = pd.DataFrame(X_array, columns=feature_names)
    y = pd.Series(y, name="Purchase")
    return X, y
