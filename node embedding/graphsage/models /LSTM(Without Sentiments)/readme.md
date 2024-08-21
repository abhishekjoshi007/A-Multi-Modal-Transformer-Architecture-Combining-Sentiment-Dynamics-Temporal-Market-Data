Stock Price Prediction with LSTM Models
This repository provides a comprehensive pipeline to predict stock price movements using Long Short-Term Memory (LSTM) models. The process involves generating labels, training models with integrated and baseline features, evaluating model performance, and calculating cosine similarity to understand the impact of related corporations' features.

Prerequisites
Before running the scripts, ensure you have the following installed:

Python 3.x
NumPy
Pandas
PyTorch
scikit-learn
tqdm
You can install the required packages using:

bash
Copy code
pip install numpy pandas torch scikit-learn tqdm
Steps to Follow
Step 1: Generate Labels
Script: label.py

This script generates the labels needed for training your models. It reads historical stock price data, calculates whether the stock price went up or down, and saves these labels to a .npy file.

Usage:

bash
Copy code
python label.py
Output:

A .npy file containing the labels that indicate stock price movement (up or down).
Step 2: Train LSTM Model with Integrated Features
Script: lstm_IF.py

This script trains the LSTM model using the integrated features generated earlier with GraphSAGE embeddings. It utilizes the labels created in Step 1 and trains the model to predict stock price movement.

Usage:

bash
Copy code
python lstm_IF.py
Output:

A trained LSTM model saved to a .pth file.
Step 3: Train LSTM Model with Baseline Features
Script: lstm_BF.py

This script trains another LSTM model, but this time using only the baseline features (i.e., features of the target company without the integration of related corporations' features). This model serves as a baseline comparison for evaluating performance.

Usage:

bash
Copy code
python lstm_BF.py
Output:

A trained LSTM model using baseline features, saved to a .pth file.
Step 4: Evaluate Models
Script: evaluate.py

This script evaluates the accuracy of both models (with integrated features and baseline features) and compares their performance. It outputs the accuracy of each model and indicates which model performed better.

Usage:

bash
Copy code
python evaluate.py
Output:

Accuracy scores for both models.
A comparison of the models' performance.


A distribution of cosine similarities.
Insights into the impact of integrating related corporations' features.
Conclusion
By following these steps, you can generate labels, train and evaluate LSTM models, and analyze the importance of related corporations' features in predicting stock price movements. The scripts are designed to be modular, allowing you to customize and extend them for your specific needs.