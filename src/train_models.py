import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Improve CPU parallel usage
os.environ["OMP_NUM_THREADS"] = "8"

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
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
train, test = load_data(
    "data/UNSW_NB15_training-set.csv",
    "data/UNSW_NB15_testing-set.csv"
)

df = pd.concat([train, test])

X, y, preprocessor = preprocess_features(df)

# -----------------------
# Pie chart BEFORE SMOTE
# -----------------------
plot_class_distribution(y, "outputs/before_smote.png")

# -----------------------
# Correlation Heatmap
# -----------------------
correlation_heatmap(df)

# -----------------------
# Train Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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
importance = feature_importance(X_train, y_train)

plt.figure(figsize=(10,5))
plt.bar(range(len(importance)), importance)
plt.title("Feature Importance (%)")
plt.savefig("outputs/feature_importance.png")
plt.close()

# -----------------------
# PCA
# -----------------------
X_train_pca, pca = apply_pca(X_train, 20)
X_test_pca = pca.transform(X_test)

# -----------------------
# Models + Parameter Grids
# -----------------------
models = {

    "Logistic": (
        LogisticRegression(max_iter=2000, n_jobs=-1),
        {"C":[0.1,1,10]}
    ),

    "KNN": (
        KNeighborsClassifier(algorithm="auto"),
        {"n_neighbors":[3,5,7]}
    ),

    "DecisionTree": (
        DecisionTreeClassifier(),
        {"max_depth":[5,10,20]}
    ),

    "RandomForest": (
        RandomForestClassifier(n_jobs=-1),
        {
            "n_estimators":[100,200],
            "max_depth":[5,10]
        }
    ),

    "SVM": (
        LinearSVC(max_iter=5000),
        {"C":[0.1,1,10]}
    ),

    "GradientBoost": (
        GradientBoostingClassifier(),
        {
            "n_estimators":[100,200],
            "learning_rate":[0.05,0.1]
        }
    ),

    "NaiveBayes": (
        GaussianNB(),
        {
            "var_smoothing":[1e-12,1e-10,1e-8]
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

    # 10 Fold Cross Validation (parallel)
    cv_scores = cross_val_score(
        model,
        X_train_pca,
        y_train,
        cv=10,
        n_jobs=-1
    )

    # GridSearchCV Hyperparameter tuning
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

    # Evaluation metrics
    acc, prec, rec, f1 = evaluate_model(y_test, y_pred)

    print("10 Fold CV Accuracy:", cv_scores.mean())
    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)

    results.append([name,cv_scores.mean(),acc,prec,rec,f1])

    # -----------------------
    # Best Model Selection
    # -----------------------
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

results_df.to_csv("outputs/model_comparison.csv",index=False)

# -----------------------
# Save Best Model
# -----------------------
joblib.dump(best_model,"models/final_model.pkl")

print(f"\nBest model selected: {best_model_name}")
print("Model saved successfully.")