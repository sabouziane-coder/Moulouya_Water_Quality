"""
Training and evaluation of ensemble learning models
for Pollution Index (PI) prediction.

Dataset: Moulouya River Basin (2014)
Authors: Sara Bouziane
Reproducibility: Fixed random seed, documented environment
"""

# ======================
# 1. Imports
# ======================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor

RANDOM_STATE = 42

# ======================
# 2. Load dataset
# ======================
df = pd.read_csv("../data/processed/Final_Pollution_Index_Results.csv")

FEATURES = ["T°C", "pH", "EC", "DO", "BOD5", "PO4", "NH4", "SO4", "NO3"]
TARGET = "PI"

X = df[FEATURES]
y = df[TARGET]

df.head()

# ======================
# 3. Train–test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

# ======================
# 4. Baseline model
# ======================
baseline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("model", Ridge(alpha=1.0))
])

baseline.fit(X_train, y_train)

# ======================
# 5. Gradient Boosting
# ======================
gb_pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))
])

gb_param_grid = {
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [3, 5, 7]
}

gb_grid = GridSearchCV(
    gb_pipeline,
    gb_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_

# ======================
# 6. AdaBoost
# ======================
ada_pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("model", AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=2),
        random_state=RANDOM_STATE
    ))
])

ada_param_grid = {
    "model__n_estimators": [50, 100, 150],
    "model__learning_rate": [0.01, 0.1, 1.0]
}

ada_grid = GridSearchCV(
    ada_pipeline,
    ada_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

ada_grid.fit(X_train, y_train)
best_ada = ada_grid.best_estimator_

# ======================
# 7. Stacking Ensemble
# ======================
stack = StackingRegressor(
    estimators=[
        ("gb", best_gb),
        ("ada", best_ada)
    ],
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

stack.fit(X_train, y_train)

# ======================
# 8. Cross-validation results
# ======================
models = {
    "Ridge Regression": baseline,
    "Gradient Boosting": best_gb,
    "AdaBoost": best_ada,
    "Stacking Ensemble": stack
}

cv_scores_dict = {}
print("\nCross-validated performance (R²):")
for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="r2"
    )
    cv_scores_dict[name] = scores
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
    
# ======================
# 9. Final test evaluation
# ======================
test_r2 = {}
test_rmse = {}

print("\nTest set performance:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    test_r2[name] = r2_score(y_test, y_pred)
    test_rmse[name] = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name}: R²={test_r2[name]:.3f}, RMSE={test_rmse[name]:.3f}")
model_names = list(test_r2.keys())
x = np.arange(len(model_names))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ---- RMSE subplot ----
axes[0].bar(model_names, test_rmse.values())
axes[0].set_ylabel("RMSE (test set)")
axes[0].set_title("(a) RMSE")
axes[0].tick_params(axis='x', rotation=90) 
# ---- R² subplot ----
axes[1].bar(model_names, test_r2.values())
axes[1].set_ylabel("R² (test set)")
axes[1].set_ylim(0, 1)
axes[1].set_title("(b) R²")
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(
    "test_performance_metrics.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# ======================
# 10. CV R² distribution plot
# ======================
plt.figure(figsize=(7, 5))
plt.boxplot(cv_scores_dict.values(), labels=cv_scores_dict.keys())
plt.ylabel("Cross-validated R²")
plt.title("")
plt.tight_layout()
plt.savefig("cross_validated_R2_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================
# 11. Scatter plots generation
# ======================
colors = {
    "Ridge Regression": "#4C72B0",      # muted blue
    "Gradient Boosting": "#55A868",     # muted green
    "AdaBoost": "#C44E52",              # muted red
    "Stacking Ensemble": "#8172B2"      # muted purple
}
y_true = y_test.values
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

min_val = min(y_true)
max_val = max(y_true)

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    
    ax.scatter(y_true, y_pred, alpha=0.7, color=colors[name])
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")
    
    
    ax.set_title(name)
    ax.set_xlabel("Observed PI")
    ax.set_ylabel("Predicted PI")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

plt.tight_layout()
plt.savefig("scatter_observed_vs_predicted_PI.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================
# 12. Ridge coefficient analysis
# ======================
ridge_estimator = baseline.named_steps["model"]
coef_series = pd.Series(
    ridge_estimator.coef_,
    index=FEATURES
).sort_values(key=abs, ascending=False)

plt.figure(figsize=(7, 4))
coef_series.plot(kind="bar", color="#4C72B0")
plt.ylabel("Coefficient magnitude")
plt.title("")
plt.tight_layout()
plt.savefig("ridge_coefficients.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================
# 13. GB feature importance
# ======================
gb_model = models["Gradient Boosting"]  

# Extract feature importances
importances = gb_model.named_steps["model"].feature_importances_
feature_names = X_train.columns  

# Sort by importance
sorted_idx = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_idx]
sorted_features = feature_names[sorted_idx]

# Plot
plt.figure(figsize=(10, 5))
plt.bar(sorted_features, sorted_importances)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Feature importance (GB)")
plt.title("")
plt.tight_layout()
plt.savefig("gb_feature_importance.png", dpi=300)
plt.show()

# ======================
# 14. Residual analysis
# ======================
y_pred = best_gb.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(5, 4))
plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted PI")
plt.ylabel("Residuals")
plt.title("Residual analysis (Gradient Boosting)")
plt.tight_layout()
plt.savefig("residuals_gradient_boosting.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================
# 15. Bootstrap procedure
# ======================
n_boot = 1000
bootstrap_r2 = []

for _ in range(n_boot):
    # Resample with replacement
    X_res, y_res = resample(X_scaled, y, replace=True)
    
    # Fit the model
    model = Ridge(alpha=1.0)
    model.fit(X_res, y_res)
    
    # Predict on the same resampled dataset (bootstrap estimation)
    y_pred = model.predict(X_res)
    
    # Compute R²
    r2 = r2_score(y_res, y_pred)
    bootstrap_r2.append(r2)

bootstrap_r2 = np.array(bootstrap_r2)


lower = np.percentile(bootstrap_r2, 2.5)
upper = np.percentile(bootstrap_r2, 97.5)
mean_r2 = np.mean(bootstrap_r2)

print("Bootstrap R² mean:", mean_r2)
print("95% CI:", lower, upper)

plt.figure(figsize=(10, 6))
plt.hist(bootstrap_r2, bins=30, edgecolor="black", alpha=0.7)
plt.axvline(mean_r2, color="red", linestyle="--", label=f"Mean R² = {mean_r2:.3f}")
plt.axvline(lower, color="green", linestyle="--", label=f"2.5% = {lower:.3f}")
plt.axvline(upper, color="green", linestyle="--", label=f"97.5% = {upper:.3f}")

plt.title("Bootstrap Distribution of R² (Ridge Regression)", fontsize=14)
plt.xlabel("R²", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# Save figure for manuscript
plt.savefig("bootstrap_R2_distribution.png", dpi=300)
plt.show()

# ======================
# 16. Seasonal variability contribution 
# ======================
FEATURES_SEASON = FEATURES + ["Season"]

X1 = df[FEATURES]
X2 = df[FEATURES_SEASON]
y = df["PI"]

# Split
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
X2_train, X2_test, _, _ = train_test_split(X2, y, test_size=0.2, random_state=42)

# Normalize
scaler1 = MinMaxScaler().fit(X1_train)
scaler2 = MinMaxScaler().fit(X2_train)

X1_train_s = scaler1.transform(X1_train)
X1_test_s = scaler1.transform(X1_test)
X2_train_s = scaler2.transform(X2_train)
X2_test_s = scaler2.transform(X2_test)

# Train the two models
ridge1 = Ridge(alpha=1.0).fit(X1_train_s, y_train)
ridge2 = Ridge(alpha=1.0).fit(X2_train_s, y_train)

# CV scores
cv1 = cross_val_score(Ridge(alpha=1.0), X1_train_s, y_train, cv=5, scoring="r2")
cv2 = cross_val_score(Ridge(alpha=1.0), X2_train_s, y_train, cv=5, scoring="r2")

print("Ridge (no season):", cv1.mean(), "±", cv1.std())
print("Ridge (+ season): ", cv2.mean(), "±", cv2.std())

# Test scores
from sklearn.metrics import r2_score

r2_test_1 = r2_score(y_test, ridge1.predict(X1_test_s))
r2_test_2 = r2_score(y_test, ridge2.predict(X2_test_s))

print("\nTest R² without season:", r2_test_1)
print("Test R² with season:    ", r2_test_2)
print("Increase:", (r2_test_2 - r2_test_1) * 100, "%")
