import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Improve CPU parallel usage
os.environ["OMP_NUM_THREADS"] = "8"

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from data_preprocessing import *
from feature_engineering import *
from evaluate import *

# -----------------------
# Create folders
# -----------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------
# Load Data
# -----------------------
train_df, test_df = load_data(
    "data/UNSW_NB15_training-set.csv",
    "data/UNSW_NB15_testing-set.csv"
)

# Separate features and labels
X_train_raw = train_df.drop("label", axis=1)
y_train = train_df["label"]

X_test_raw = test_df.drop("label", axis=1)
y_test = test_df["label"]

# -----------------------
# Class distribution BEFORE SMOTE
# -----------------------
plot_class_distribution(y_train, "outputs/before_smote.png")

# -----------------------
# Correlation Heatmap
# -----------------------
correlation_heatmap(train_df)

# -----------------------
# Preprocessing
# -----------------------
X_train, X_test, preprocessor = preprocess_features(
    X_train_raw,
    X_test_raw
)

# -----------------------
# SMOTE
# -----------------------
X_train, y_train = apply_smote(X_train, y_train)

plot_class_distribution(y_train, "outputs/after_smote.png")

# -----------------------
# Scaling
# -----------------------
X_train, X_test, scaler = scale_features(X_train, X_test)

# -----------------------
# Feature Importance
# -----------------------

# Calculate feature importance
importance = feature_importance(X_train, y_train)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Convert to numpy array
feature_names = np.array(feature_names)

# Get top 20 most important features
indices = np.argsort(importance)[-20:]

plt.figure(figsize=(12,6))
plt.barh(feature_names[indices], importance[indices])

plt.xlabel("Importance (%)")
plt.title("Top 20 Most Important Features")

plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# -----------------------
# PCA
# -----------------------
X_train_pca, X_test_pca, pca = apply_pca(
    X_train,
    X_test,
    n_components=20
)

# -----------------------
# Models
# -----------------------
models = {

    "Logistic": (
        LogisticRegression(max_iter=2000, n_jobs=-1),
        {"C":[0.1,1]}
    ),

    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors":[3,5]}
    ),

    "DecisionTree": (
        DecisionTreeClassifier(),
        {"max_depth":[10,20]}
    ),

    "RandomForest": (
        RandomForestClassifier(n_jobs=-1),
        {
            "n_estimators":[100,200],
            "max_depth":[5,10]
        }
    ),

    "SVM_RBF": (
        SVC(kernel="rbf"),
        {"C":[0.1,1], "gamma":["scale","auto"]}
    ),

    "NaiveBayes": (
        GaussianNB(),
        {
            "var_smoothing":[1e-12,1e-10]
        }
    )
}

results = []

best_f1 = 0
best_recall = 0
best_accuracy = 0
best_model = None
best_model_name = ""

# -----------------------
# Train Models
# -----------------------
for name,(model,param_grid) in models.items():

    print(f"\n{name} Model")

    # 10 Fold Cross Validation
    cv_scores = cross_val_score(
        model,
        X_train_pca,
        y_train,
        cv=10,
        n_jobs=-1
    )

    # GridSearch
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train_pca, y_train)

    tuned_model = grid.best_estimator_

    # Prediction
    y_pred = tuned_model.predict(X_test_pca)

    # Evaluation
    acc, prec, rec, f1 = evaluate_model(y_test, y_pred)

    print("10 Fold CV Accuracy:", cv_scores.mean())
    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)

    results.append([name,cv_scores.mean(),acc,prec,rec,f1])

    # Best Model Selection
    if (
        f1 > best_f1 or
        (f1 == best_f1 and rec > best_recall) or
        (f1 == best_f1 and rec == best_recall and acc > best_accuracy)
    ):
        best_f1 = f1
        best_recall = rec
        best_accuracy = acc
        best_model = tuned_model
        best_model_name = name


# -----------------------
# Save comparison
# -----------------------
results_df = pd.DataFrame(
    results,
    columns=["Model","CV Accuracy","Accuracy","Precision","Recall","F1"]
)

results_df.to_csv("data/model_comparison.csv",index=False)

# -----------------------
# Model Comparison Bar Graph
# -----------------------
plt.figure(figsize=(10,6))

x = range(len(results_df["Model"]))

plt.bar(x, results_df["Accuracy"], width=0.2, label="Accuracy")
plt.bar([i + 0.2 for i in x], results_df["Precision"], width=0.2, label="Precision")
plt.bar([i + 0.4 for i in x], results_df["Recall"], width=0.2, label="Recall")
plt.bar([i + 0.6 for i in x], results_df["F1"], width=0.2, label="F1")

plt.xticks([i + 0.3 for i in x], results_df["Model"], rotation=45)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# -----------------------
# Save Best Model,preprocessor,scaler, pca using Joblib
# -----------------------
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(preprocessor,"models/preprocessor.pkl")
joblib.dump(scaler,"models/scaler.pkl")
joblib.dump(pca,"models/pca.pkl")

print(f"\nBest model selected: {best_model_name}")
print("Model saved successfully as models/best_model.pkl")