import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


# -----------------------------
# Feature Importance
# -----------------------------
def feature_importance(X_train, y_train):

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    importance = model.feature_importances_

    importance_percent = importance * 100

    return importance_percent


# -----------------------------
# Correlation Heatmap
# -----------------------------
def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=(12,8))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        annot=False
    )

    plt.title("Correlation Heatmap")

    plt.savefig("outputs/correlation_heatmap.png")

    plt.close()


# -----------------------------
# PCA Dimensionality Reduction
# -----------------------------
def apply_pca(X_train, X_test, n_components=20):

    pca = PCA(n_components=n_components)

    # Fit on train only
    X_train_pca = pca.fit_transform(X_train)

    # Transform test
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca