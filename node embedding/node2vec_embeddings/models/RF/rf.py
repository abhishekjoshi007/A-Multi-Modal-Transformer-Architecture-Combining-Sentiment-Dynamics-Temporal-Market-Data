import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define paths to Node2Vec features and labels
node2vec_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/integrated_features'
labels_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/node2vec_embeddings/models/LSTM (without sentiments)/labels.npy'

# Load the tickers (to match features and labels)
tickers = [f.replace('_features.npy', '') for f in os.listdir(node2vec_features_dir) if f.endswith('.npy')]

# Initialize lists to store features and labels
features = []
labels = np.load(labels_path)

# Load features
for ticker in tickers:
    feature_path = os.path.join(node2vec_features_dir, f"{ticker}_features.npy")
    if os.path.exists(feature_path):
        features.append(np.load(feature_path))
    else:
        print(f"Feature file for {ticker} not found!")

# Ensure that features and labels are correctly aligned
if len(features) != len(labels):
    print("Mismatch between the number of features and labels!")
    print(f"Number of features: {len(features)}")
    print(f"Number of labels: {len(labels)}")
    # Optionally, handle mismatch here
else:
    print("Features and labels are correctly aligned.")

# Convert features to a numpy array and flatten if necessary
features = np.array([feature.flatten() for feature in features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the model with the best hyperparameters
best_rf_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Random Forest with best hyperparameters: {accuracy * 100:.2f}%")

# Feature importance
importances = best_rf_model.feature_importances_
for i, importance in enumerate(importances):
    print(f"Feature {i + 1}: {importance:.4f}")

# Cross-validation to check model consistency
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean() * 100:.2f}%")
