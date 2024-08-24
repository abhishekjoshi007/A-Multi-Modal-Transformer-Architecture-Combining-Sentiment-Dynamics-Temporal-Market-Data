import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load your GraphSAGE embeddings and labels
graphsage_features_dir = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/integrated_features'
labels_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/models /RF/corrected_labels.npy'

# Load tickers to match features and labels
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

# Ensure features and labels are aligned
if len(features) != len(labels):
    print("Mismatch between features and labels!")
    print(f"Number of features: {len(features)}")
    print(f"Number of labels: {len(labels)}")
else:
    print("Features and labels are correctly aligned.")

# Convert features to numpy array
features = np.array(features)

# Split data into training, validation, and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Build the StellarGraph object for GraphSAGE
node_features_train = pd.DataFrame(X_train, index=[f"node_{i}" for i in range(X_train.shape[0])])
node_features_val = pd.DataFrame(X_val, index=[f"node_{i+len(X_train)}" for i in range(X_val.shape[0])])
node_features_test = pd.DataFrame(X_test, index=[f"node_{i+len(X_train)+len(X_val)}" for i in range(X_test.shape[0])])

graph_train = StellarGraph(nodes=node_features_train)
graph_val = StellarGraph(nodes=node_features_val)
graph_test = StellarGraph(nodes=node_features_test)

# Create the GraphSAGE generator
batch_size = 32
num_samples = [10, 5]
generator_train = GraphSAGENodeGenerator(graph_train, batch_size=batch_size, num_samples=num_samples)
generator_val = GraphSAGENodeGenerator(graph_val, batch_size=batch_size, num_samples=num_samples)
generator_test = GraphSAGENodeGenerator(graph_test, batch_size=batch_size, num_samples=num_samples)

# Define a function to create the GraphSAGE model
def create_graphsage_model(layer_sizes=[32, 32], dropout=0.5, l2_reg=0.001, l1_reg=0.0, learning_rate=0.01):
    graphsage_model = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator_train, bias=True, dropout=dropout,
        kernel_regularizer=l2(l2_reg) if l1_reg == 0 else l1(l1_reg)
    )
    
    x_inp, x_out = graphsage_model.in_out_tensors()
    
    # Batch Normalization
    x_out = BatchNormalization()(x_out)
    
    # Dense layer with dropout
    x_out = Dense(128, activation="relu")(x_out)
    x_out = Dropout(dropout)(x_out)
    
    # Final prediction layer
    prediction = Dense(units=1, activation="sigmoid")(x_out)
    
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

# Example hyperparameter grid
param_grid = {
    'layer_sizes': [[32, 32], [64, 64], [128, 128]],
    'dropout': [0.3, 0.5, 0.7],
    'l2_reg': [0.001, 0.01],
    'l1_reg': [0.0, 0.001],
    'learning_rate': [0.001, 0.01, 0.1],
}

best_accuracy = 0
best_params = {}

# K-Fold Cross-Validation
kf = KFold(n_splits=5)
cumulative_accuracy = 0

for train_index, val_index in kf.split(X_train_full):
    X_train_cv, X_val_cv = X_train_full[train_index], X_train_full[val_index]
    y_train_cv, y_val_cv = y_train_full[train_index], y_train_full[val_index]
    
    node_features_train_cv = pd.DataFrame(X_train_cv, index=[f"node_{i}" for i in range(X_train_cv.shape[0])])
    node_features_val_cv = pd.DataFrame(X_val_cv, index=[f"node_{i+len(X_train_cv)}" for i in range(X_val_cv.shape[0])])
    
    graph_train_cv = StellarGraph(nodes=node_features_train_cv)
    graph_val_cv = StellarGraph(nodes=node_features_val_cv)
    
    generator_train_cv = GraphSAGENodeGenerator(graph_train_cv, batch_size=batch_size, num_samples=num_samples)
    generator_val_cv = GraphSAGENodeGenerator(graph_val_cv, batch_size=batch_size, num_samples=num_samples)
    
    for layer_sizes in param_grid['layer_sizes']:
        for dropout in param_grid['dropout']:
            for l2_reg in param_grid['l2_reg']:
                for l1_reg in param_grid['l1_reg']:
                    for learning_rate in param_grid['learning_rate']:
                        
                        model = create_graphsage_model(layer_sizes, dropout, l2_reg, l1_reg, learning_rate)
                        train_gen = generator_train_cv.flow(node_features_train_cv.index, y_train_cv, shuffle=True)
                        val_gen = generator_val_cv.flow(node_features_val_cv.index, y_val_cv)

                        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
                        
                        model.fit(train_gen, epochs=50, validation_data=val_gen, callbacks=[early_stopping, lr_scheduler], verbose=2)
                        
                        val_gen_test = generator_val_cv.flow(node_features_val_cv.index, y_val_cv)
                        accuracy = model.evaluate(val_gen_test)[1]
                        
                        print(f"Accuracy with layer sizes {layer_sizes}, dropout {dropout}, l2_reg {l2_reg}, l1_reg {l1_reg}, learning_rate {learning_rate}: {accuracy * 100:.2f}%")
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'layer_sizes': layer_sizes,
                                'dropout': dropout,
                                'l2_reg': l2_reg,
                                'l1_reg': l1_reg,
                                'learning_rate': learning_rate
                            }
                        
                        cumulative_accuracy += accuracy

print(f"Best hyperparameters: {best_params}, Best accuracy: {best_accuracy * 100:.2f}%")

# Print cumulative accuracy
print(f"Cumulative Accuracy over all folds and hyperparameter combinations: {cumulative_accuracy / (5 * len(param_grid['layer_sizes']) * len(param_grid['dropout']) * len(param_grid['l2_reg']) * len(param_grid['l1_reg']) * len(param_grid['learning_rate'])) * 100:.2f}%")

# Final model training with best hyperparameters on full training data
final_model = create_graphsage_model(**best_params)
train_gen = generator_train.flow(node_features_train.index, y_train, shuffle=True)
val_gen = generator_val.flow(node_features_val.index, y_val)
test_gen = generator_test.flow(node_features_test.index, y_test)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

final_model.fit(train_gen, epochs=50, validation_data=val_gen, callbacks=[early_stopping, lr_scheduler], verbose=2)

# Evaluate the model on the test set
test_accuracy = final_model.evaluate(test_gen)[1]
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Evaluate the model on additional metrics
y_pred = final_model.predict(test_gen).ravel()

precision = precision_score(y_test, y_pred.round())
recall = recall_score(y_test, y_pred.round())
f1 = f1_score(y_test, y_pred.round())
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")
