# Machine Learning Assignment 2

## a. Problem Statement
The objective of this project is to implement multiple machine learning classification models to predict whether a breast cancer tumor is benign or malignant. We evaluate and compare the models' performance using various metrics (Accuracy, AUC, Precision, Recall, F1, MCC) and deploy an interactive web application using Streamlit to demonstrate the predictions on test data.

## b. Dataset Description
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, obtained via `scikit-learn` (originally from the UCI Machine Learning Repository).
*   **Number of Instances:** 569 (Meets the minimum requirement of 500)
*   **Number of Features:** 30 numeric, predictive attributes (Meets the minimum requirement of 12)
*   **Target:** Binary classification (0 = Malignant, 1 = Benign)
Features include mean radius, mean texture, mean perimeter, mean area, mean smoothness, and other computed metrics of cell nuclei from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## c. Github Repository Link
[Insert your GitHub Repository Link here]

## d. Models Used
The following 5 machine learning models were trained on 80% of the dataset and evaluated on the remaining 20% test data.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9561 | 0.9977 | 0.9459 | 0.9859 | 0.9655 | 0.9068 |
| Decision Tree | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| kNN | 0.9561 | 0.9959 | 0.9342 | 1.0000 | 0.9660 | 0.9086 |
| Naive Bayes | 0.9737 | 0.9984 | 0.9595 | 1.0000 | 0.9793 | 0.9447 |
| Random Forest (Ensemble) | 0.9649 | 0.9953 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |

## e. Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performs very well with high Accuracy (95.6%) and AUC (99.7%). It effectively models the linear relationship between the features and the target. |
| Decision Tree | Shows slightly lower accuracy and AUC compared to others, likely due to overfitting the training data, but still performs decently (94.7% accuracy). |
| kNN | Excellent recall (100%) but slightly lower precision. It successfully identifies all benign cases in the test set but has some false positives. |
| Naive Bayes | **Best performer** among simple models for this split, with 97.3% accuracy, perfect recall, and the highest MCC (0.944), indicating strong predictive power despite the independence assumption. |
| Random Forest (Ensemble) | Very robust performance (96.4% accuracy). By aggregating multiple decision trees, it mitigates the overfitting seen in the single Decision Tree model. |
| **Overall Winner for your dataset?** | **Naive Bayes** and **Random Forest** are the top performers. Naive Bayes edges out slightly on this specific test set with the highest F1 and MCC scores. |

## f. Code Workflow & Architecture

The machine learning pipeline is implemented across 5 separate model scripts in the `model/` directory (e.g., `knn.py`, `decision_tree.py`). Each script follows a standardized, 5-step workflow:

### 1. Data Loading and Preparation
- The breast cancer dataset is loaded directly using `scikit-learn` (`load_breast_cancer()`).
- The data is split into 80% training data (`X_train`, `y_train`) and 20% testing data (`X_test`, `y_test`) using `train_test_split` with a fixed `random_state=42` to ensure consistent evaluation across all models.
- A copy of the test data is exported and saved to `test_data.csv` for use in the Streamlit application.

### 2. Model Initialization and Training
- The specific machine learning algorithm (e.g., `KNeighborsClassifier`) is initialized from `scikit-learn`.
- The model is trained on the 80% training data using the `.fit(X_train, y_train)` method, allowing the algorithm to learn the mathematical patterns that distinguish benign from malignant tumors.

### 3. Generating Predictions
- The trained model evaluates the 20% unseen testing data using `.predict(X_test)` to generate categorical predictions (0 or 1).
- It also uses `.predict_proba(X_test)` to generate confidence probabilities, which are required for calculating the AUC (Area Under the ROC Curve) metric.

### 4. Evaluating the Model
- The model's predictions are compared against the actual true labels (`y_test`) that were held back during training.
- We utilize `scikit-learn`'s built-in metric functions (e.g., `accuracy_score`, `roc_auc_score`) to automatically calculate the assignment's required metrics: Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

### 5. Saving the Model Weights
- Instead of discarding the trained model from memory, we serialize the learned weights and parameters using `joblib.dump()`.
- The weights are saved as `.joblib` files directly in the `model/` directory. 
- **Deployment Benefit:** This architecture allows our Streamlit web application (`app.py`) to instantly load the pre-trained weights to make predictions on user-uploaded test data, eliminating the need to retrain the models every time the webpage is refreshed.
