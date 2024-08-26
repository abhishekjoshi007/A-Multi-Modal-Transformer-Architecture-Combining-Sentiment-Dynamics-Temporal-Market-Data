# Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Function to Load GraphSAGE Embeddings from a CSV File
def load_graphsage_embeddings(embeddings_file):
    return pd.read_csv(embeddings_file).values

# Function to Load Labels
def load_labels(labels_file):
    return np.load(labels_file)

# Load Data
embeddings_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/embeddings.csv'  # Replace with your actual path
labels_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/models /RF/corrected_labels.npy'  # Replace with your actual path

graphsage_embeddings = load_graphsage_embeddings(embeddings_file)
labels = load_labels(labels_file)

# Check for consistency in data size
if graphsage_embeddings.shape[0] != len(labels):
    min_samples = min(graphsage_embeddings.shape[0], len(labels))
    graphsage_embeddings = graphsage_embeddings[:min_samples]
    labels = labels[:min_samples]
    print(f"Adjusted embeddings and labels to {min_samples} samples for consistency.")

# Feature Scaling
scaler = StandardScaler()
graphsage_embeddings = scaler.fit_transform(graphsage_embeddings)

# Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(graphsage_embeddings, labels)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Expanded Grid Search for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=rf_param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Gradient Boosting with Parameter Tuning
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}

grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                              param_grid=gb_param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_gb.fit(X_train, y_train)
best_gb_model = grid_search_gb.best_estimator_

# XGBoost Model
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Neural Network Model with Further Adjustments
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42, solver='lbfgs', learning_rate_init=0.001, alpha=0.0001)
mlp_model.fit(X_train, y_train)

# Ensemble with Stacking
estimators = [
    ('rf', best_rf_model),
    ('gb', best_gb_model),
    ('xgb', xgb_model),
    ('mlp', mlp_model)
]
stacked_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
stacked_model.fit(X_train, y_train)

# Predictions and Evaluation for Stacked Model
y_pred_stacked = stacked_model.predict(X_test)
accuracy_stacked = accuracy_score(y_test, y_pred_stacked)
f1_stacked = f1_score(y_test, y_pred_stacked, zero_division=0)
precision_stacked = precision_score(y_test, y_pred_stacked, zero_division=0)
recall_stacked = recall_score(y_test, y_pred_stacked, zero_division=0)
roc_auc_stacked = roc_auc_score(y_test, stacked_model.predict_proba(X_test)[:, 1])

print(f"Stacked Model - Accuracy: {accuracy_stacked:.4f}")
print(f"Stacked Model - F1 Score: {f1_stacked:.4f}")
print(f"Stacked Model - Precision: {precision_stacked:.4f}")
print(f"Stacked Model - Recall: {recall_stacked:.4f}")
print(f"Stacked Model - ROC-AUC: {roc_auc_stacked:.4f}")
print(classification_report(y_test, y_pred_stacked))

# Save the Stacked Model
joblib.dump(stacked_model, 'stacked_model.pkl')
print("Stacked model saved as stacked_model.pkl")
