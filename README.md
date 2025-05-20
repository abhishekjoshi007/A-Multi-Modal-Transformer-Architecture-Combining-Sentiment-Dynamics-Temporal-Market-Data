# A Multi-Modal Transformer Architecture Combining Sentiment Dynamics, Temporal Market Data, and Macroeconomic Indicators for Sturdy Stock Return Forecasting

This repository contains the official implementation of our IEEE Big Data 2024 paper:

**"A Multi-Modal Transformer Architecture Combining Sentiment Dynamics, Temporal Market Data, and Macroeconomic Indicators for Sturdy Stock Return Forecasting"**  
by **Abhishek Joshi**, **Jahnavi Krishna Koda**, and **Alihan Hadimlioglu**

📄 **[IEEE Paper Link](https://ieeexplore.ieee.org/document/10825219)**  
🔗 **DOI:** [10.1109/BigData62323.2024.10825219](https://doi.org/10.1109/BigData62323.2024.10825219)


## 🧠 Overview

This project presents a novel multi-modal forecasting framework that integrates:

- **Volume-weighted sentiment signals** from Reddit and Yahoo Finance discussions
- **Temporal market indicators** like OHLCV and volatility
- **Macroeconomic indicators** including inflation, GDP, and interest rates

We propose an adaptive Transformer-based architecture capable of robustly predicting **next-day stock returns** by dynamically learning from multimodal inputs. The system also integrates **Graph Neural Networks** to model inter-stock relationships and **LSTM & Random Forest** as comparative baselines.

## 🚀 Getting Started

### 1. Clone the repository
git clone [https://github.com/yourusername/multimodal-stock-prediction.git](https://github.com/abhishekjoshi007/A-Multi-Modal-Transformer-Architecture-Combining-Sentiment-Dynamics-Temporal-Market-Data)

cd multimodal-stock-prediction


### 2. Set up the environment
conda create -n stockpred python=3.10
conda activate stockpred
pip install -r requirements.txt

### 3. Prepare the data
* Place historical stock data (Yahoo Finance)
* Run `python utils/preprocessing.py` to process sentiment and macro data

### 4. Train the model
python train.py --model transformer --config config.yaml


## 📌 Citation

If you use this code or reference this work, please cite:

bibtex

@INPROCEEDINGS{10825219,
  author={Joshi, Abhishek and Koda, Jahnavi Krishna and Hadimlioglu, Alihan},
  booktitle={2024 IEEE International Conference on Big Data (BigData)}, 
  title={A Multi-Modal Transformer Architecture Combining Sentiment Dynamics, Temporal Market Data, and Macroeconomic Indicators for Sturdy Stock Return Forecasting}, 
  year={2024},
  pages={4896-4902},
  doi={10.1109/BigData62323.2024.10825219}
}

## 📬 Contact

For questions, collaborations, or feedback, feel free to connect on LinkedIn- https://www.linkedin.com/in/abhishek-joshi-510b68151/ / Mail - abhishek.07joshi@gmail.com or open an issue in this repository.
