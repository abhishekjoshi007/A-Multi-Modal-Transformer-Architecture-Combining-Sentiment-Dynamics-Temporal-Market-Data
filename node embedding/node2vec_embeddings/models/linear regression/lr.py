# Import Required Libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score  # Make sure to import cross_val_score
from imblearn.over_sampling import SMOTE
import os
import joblib

# Function to Load Node2Vec Embeddings for All Dates (Handling Non-Numerical Data)
def load_all_node2vec_embeddings(embeddings_folder):
    embeddings_dict = {}
    for date_folder in sorted(os.listdir(embeddings_folder)):
        embeddings_file = os.path.join(embeddings_folder, date_folder, f'{date_folder}_embeddings.emb')
        if os.path.isfile(embeddings_file):
            with open(embeddings_file, 'r') as f:
                embeddings = []
                for line in f:
                    parts = line.strip().split()
                    # Skip non-numerical data
                    if len(parts) > 1 and parts[0].replace('.', '', 1).isdigit():
                        vector = np.array(list(map(float, parts[1:])))
                        embeddings.append(vector)
                embeddings_dict[date_folder] = np.array(embeddings)
    return embeddings_dict

# Function to Load Labels
def load_labels(labels_file):
    return np.load(labels_file)

# Load Data
embeddings_folder = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/organized_embeddings'  # Replace with your actual path
labels_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/models/LSTM (without sentiments)/labels.npy'  # Replace with your actual path

node2vec_embeddings = load_all_node2vec_embeddings(embeddings_folder)
labels = load_labels(labels_file)

# Flatten embeddings and labels into single arrays for training
node2vec_data = []
all_labels = []

for date in sorted(node2vec_embeddings.keys()):
    node2vec_data.extend(node2vec_embeddings[date])
    all_labels.extend(labels[:node2vec_embeddings[date].shape[0]])
    labels = labels[node2vec_embeddings[date].shape[0]:]  # Update labels

node2vec_data = np.array(node2vec_data)
all_labels = np.array(all_labels)

# Feature Scaling
scaler = StandardScaler()
node2vec_data = scaler.fit_transform(node2vec_data)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(node2vec_data, all_labels, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTE: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

# Train Logistic Regression Model on Node2Vec Embeddings with class weights
log_reg_model = LogisticRegression(class_weight='balanced', max_iter=1000)
log_reg_model.fit(X_train_res, y_train_res)

# Make Predictions and Evaluate
y_pred = log_reg_model.predict(X_test)

# Evaluate using different metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, log_reg_model.predict_proba(X_test)[:, 1])

print(f"Node2Vec Embeddings - Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Node2Vec Embeddings - Logistic Regression F1 Score: {f1:.4f}")
print(f"Node2Vec Embeddings - Logistic Regression Precision: {precision:.4f}")
print(f"Node2Vec Embeddings - Logistic Regression Recall: {recall:.4f}")
print(f"Node2Vec Embeddings - Logistic Regression ROC-AUC: {roc_auc:.4f}")

# Cross-validation Score using accuracy
cross_val_scores = cross_val_score(log_reg_model, X_train_res, y_train_res, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cross_val_scores):.4f} +/- {np.std(cross_val_scores):.4f}")

# Save the Model
joblib.dump(log_reg_model, 'logistic_regression_node2vec_model.pkl')
print("Model saved as logistic_regression_node2vec_model.pkl")
