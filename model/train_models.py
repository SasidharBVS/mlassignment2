import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
print("Loading dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for Streamlit app
print("Saving test_data.csv...")
test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv('test_data.csv', index=False)

# 3. Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'kNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest (Ensemble)': RandomForestClassifier(random_state=42)
}

# 4. Train, evaluate, and save models
results = []

print("Training models...")
for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test) # fallback if needed
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        'ML Model Name': name,
        'Accuracy': acc,
        'AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'MCC': mcc
    })
    
    # Save model
    filename = f"model/{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
    joblib.dump(model, filename)
    print(f"Saved {name} to {filename}")

# 5. Print Comparison Table
results_df = pd.DataFrame(results)
print("\n--- Model Comparison Table ---")
print("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
print("|---|---|---|---|---|---|---|")
for idx, row in results_df.iterrows():
    print(f"| {row['ML Model Name']} | {row['Accuracy']:.4f} | {row['AUC']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['MCC']:.4f} |")

print("\nTraining completed successfully.")

print("\nTraining completed successfully.")
