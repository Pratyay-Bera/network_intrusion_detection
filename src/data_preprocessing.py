import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


# -----------------------------
# Load Dataset
# -----------------------------
def load_data(train_path, test_path):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


# -----------------------------
# Preprocess Features
# -----------------------------
def preprocess_features(X_train, X_test):

    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, preprocessor


# -----------------------------
# Plot Class Distribution
# -----------------------------
def plot_class_distribution(y, filename):

    counts = pd.Series(y).value_counts()

    plt.figure()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.title("Class Distribution")

    plt.savefig(filename)
    plt.close()


# -----------------------------
# Apply SMOTE (TRAIN ONLY)
# -----------------------------
def apply_smote(X_train, y_train):

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


# -----------------------------
# Feature Scaling
# -----------------------------
def scale_features(X_train, X_test):

    scaler = StandardScaler(with_mean=False)

    # Fit on train only
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler




