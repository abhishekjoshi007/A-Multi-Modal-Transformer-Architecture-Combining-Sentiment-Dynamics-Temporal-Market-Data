import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define paths to GraphSAGE features and labels
graphsage_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/integrated_features'  # Update with your actual path
labels_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/models /RF/corrected_labels.npy'  # Update with your actual path

# Load the tickers (to match features and labels)
tickers = [f.replace('_features.npy', '') for f in os.listdir(graphsage_features_dir) if f.endswith('.npy')]

# Initialize lists to store features and labels
features = []
labels = np.load(labels_path)

# Load features
for ticker in tickers:
    feature_path = os.path.join(graphsage_features_dir, f"{ticker}_features.npy")
    if os.path.exists(feature_path):
        features.append(np.load(feature_path))
    else:
        print(f"Feature file for {ticker} not found!")

# Ensure that features and labels are correctly aligned
if len(features) != len(labels):
    print("Mismatch between the number of features and labels!")
    print(f"Number of features: {len(features)}")
    print(f"Number of labels: {len(labels)}")
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
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Predict on the test set with the best model
y_pred = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Random Forest with best hyperparameters: {accuracy * 100:.2f}%")

# Feature importance analysis
feature_importances = grid_search.best_estimator_.feature_importances_
print("Feature importances:")
for i, importance in enumerate(feature_importances):
    print(f"Feature {i + 1}: {importance:.4f}")

# Cross-validation to evaluate the model
cv_scores = cross_val_score(grid_search.best_estimator_, features, labels, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {np.mean(cv_scores) * 100:.2f}%")
