import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#load

data = fetch_openml('telco-customer-churn', version=1, as_frame=True)
df = data.frame.copy()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
if 'customerID' in df.columns:
    df = df.dropna(columns=['customerID'])

y = (df['Churn']== 'Yes').astype(int)
X = pd.get_dummies(df.drop(columns=['Churn']),drop_first=True)

print("Shape:",X.shape,"| Positives:", y.sum(),"/", len(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Single Decision Tree

# unconstrained overfits hard

tree_unpruned = DecisionTreeClassifier(random_state=42).fit(X_train,y_train)
print("\n[Tree unpruned]")
print(f" Train:{accuracy_score(y_train, tree_unpruned.predict(X_train)):.4f}")
print(f" Test:{accuracy_score(y_test, tree_unpruned.predict(X_test)):.4f}"  )
print(f" Depth:{tree_unpruned.get_depth()}"  )

# constrained -> bias-variance dial

tree_pruned = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42).fit(X_train,y_train)
print("\n[Tree max_depth=5, min_leaf=20]")
print(f" Train:{accuracy_score(y_train, tree_pruned.predict(X_train)):.4f}")
print(f" Test:{accuracy_score(y_test, tree_pruned.predict(X_test)):.4f}"  )
    

# Sweep max_depth to visualize the bias-vari9ance curve

depths = range(1,21)
train_accs, test_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, m.predict(X_train)))
    test_accs.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(8,5))
plt.plot(depths, train_accs, marker='o', label=('Train'))
plt.plot(depths, test_accs, marker='o', label=('Test'))
plt.xlabel("Max Depth");plt.ylabel("Accuracy")
plt.title("Decision Tree -- bias/variance vs depth")
plt.legend(); plt.grid(alpha=0.3); plt.show()


# Random Forest

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=42
).fit(X_train, y_train)

print("\n[Random Forest --]")
print(f" Train: {accuracy_score(y_train, rf.predict(X_train)):.4f}")
print(f" Test: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# XGBoost

xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, 
    subsample=0.8,colsample_bytree=0.8, eval_metric='logloss',
      early_stopping_rounds=30, random_state=42
)

xgb.fit(X_train, y_train,
eval_set=[(X_train, y_train), (X_test, y_test)],
verbose=False)

print("\n[XGBoost -- 500 rounds, lr = 0.05, early stop]")

print(f" Train:{accuracy_score(y_train, xgb.predict(X_train)):.4f}")
print(f" Test:{accuracy_score(y_test, xgb.predict(X_test)):.4f}")
print(f" Best iteration: {xgb.best_iteration}")

# XGB train/test curve vs boosting round
results = xgb.evals_result()
plt.figure(figsize=(8,5))
plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Test')
plt.axvline(xgb.best_iteration, color='red', linestyle='--', label='Early stop')
plt.xlabel("Boosting round"); plt.ylabel("Log loss")
plt.title("XGBoost -- overfitting as boosting proceeds")
plt.legend(); plt.grid(alpha=0.3); plt.show()

# summary

print("\n-- Test accuracy summary ---")
models = {
    'Tree (unpruned)': tree_unpruned,
    'Tree (depth=5)': tree_pruned,  
    'Random Forest': rf,
    'XGBoost': xgb

}

for name, m in models.items():
    print(f"{name:20s} | Test accuracy: {accuracy_score(y_test, m.predict(X_test)):.4f}")

# fearure importance comparison

def top_importances(model, n=15):
    imp = pd.Series(model.feature_importances_, index=X.columns)
    return imp.sort_values(ascending=False).head(n)

fig, axes = plt.subplots(1,3, figsize=(18,6))
for ax, (name, m) in zip(axes, [('Decision Tree', tree_pruned),
                                 ('Random Forest', rf),
                                 ('XGBoost', xgb)]):
    top_importances(m).iloc[::-1].plot(kind='barh', ax=ax)
    ax.set_title(name); ax.set_xlabel("Importance")
    plt.tight_layout(); plt.show()

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------- 7. Stacking ----------
# Base learners: intentionally diverse (different inductive biases)
estimators = [
    ('rf',  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=42)),
    ('lr',  LogisticRegression(max_iter=2000)),
]

# stacking


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


lr_scaled = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000))
])

estimators = [
    ('rf',  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=42)),
    ('lr',  lr_scaled),
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000),  
    cv=5,
    n_jobs=-1,
)
stack.fit(X_train, y_train)

print("\n[Stacking — RF + XGB + LR → LR meta]")
print(f"  Train: {accuracy_score(y_train, stack.predict(X_train)):.4f}")
print(f"  Test : {accuracy_score(y_test,  stack.predict(X_test)):.4f}")

# Full summary including stack
print("\n--- Final test accuracy summary ---")
final_models = {
    'Tree (unpruned)': tree_unpruned,
    'Tree (depth=5)':  tree_pruned,
    'Random Forest':   rf,
    'XGBoost':         xgb,
    'Stack (RF+XGB+LR)': stack,
}
for name, m in final_models.items():
    print(f"  {name:22s} {accuracy_score(y_test, m.predict(X_test)):.4f}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# ---------- 8. Confusion matrices ----------
final_models = {
    'Tree (depth=5)':  tree_pruned,
    'Random Forest':   rf,
    'XGBoost':         xgb,
    'Stack (RF+XGB+LR)': stack,
}

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, (name, m) in zip(axes, final_models.items()):
    cm = confusion_matrix(y_test, m.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn']).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()

# ---------- 9. Per-model classification report ----------
for name, m in final_models.items():
    print(f"\n=== {name} ===")
    print(classification_report(y_test, m.predict(X_test),
                                target_names=['No Churn', 'Churn'], digits=3))