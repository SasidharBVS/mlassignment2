import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Set page config
st.set_page_config(page_title="ML Model Evaluator", layout="wide")

st.title("Machine Learning Classification Model Evaluator")
st.write("""
This app evaluates various classification models on the Breast Cancer Wisconsin (Diagnostic) dataset.
Please upload the `test_data.csv` file to see the evaluation metrics and confusion matrix.
""")

# Sidebar
st.sidebar.header("User Inputs")

# Model Selection
model_options = [
    'Logistic Regression',
    'Decision Tree',
    'kNN',
    'Naive Bayes',
    'Random Forest (Ensemble)'
]
selected_model_name = st.sidebar.selectbox("Select a Model", model_options)

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

# Load the selected model
@st.cache_resource
def load_model(model_name):
    filename = f"model/{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
    return joblib.load(filename)

try:
    model = load_model(selected_model_name)
    st.sidebar.success(f"{selected_model_name} loaded successfully!")
except Exception as e:
    st.error(f"Error loading model {selected_model_name}: {e}")
    model = None

# Evaluation
if uploaded_file is not None and model is not None:
    # Read the data
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Test Data Preview")
        st.dataframe(df.head())
        
        if 'target' not in df.columns:
            st.error("The uploaded CSV must contain a 'target' column for evaluation.")
        else:
            X_test = df.drop(columns=['target'])
            y_test = df['target']
            
            # Predict
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = [0] * len(y_test)
                
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if hasattr(model, "predict_proba") else "N/A"
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.write("### Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}" if isinstance(auc, float) else auc)
            col3.metric("Precision", f"{prec:.4f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")
            
            # Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'Confusion Matrix - {selected_model_name}')
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload the test data CSV file from the sidebar to proceed.")
