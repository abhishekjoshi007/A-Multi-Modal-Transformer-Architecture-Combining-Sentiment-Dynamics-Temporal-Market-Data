# Stock Node Embedding and Similarity Analysis

This project focuses on generating node embeddings for a collection of stocks, mapping tickers to indices, calculating similarities between all nodes (stocks), and identifying the top 3 most similar stocks.

## Table of Contents

1. [Generating Node Embeddings](#generating-node-embeddings)
2. [Ticker Mapping](#ticker-mapping)
3. [Calculating Similarities](#calculating-similarities)
4. [Extracting Top 3 Similar Stocks](#extracting-top-3-similar-stocks)

## Generating Node Embeddings

### Overview
Node embeddings are vector representations of nodes in a graph, capturing the structural information of each node relative to its neighbors. In this project, we use the GraphSAGE algorithm implemented in PyTorch Geometric to generate embeddings for each stock.

### Steps
1. **Load Graph Data:**
   - Graph data is stored in GraphML format, with nodes representing stocks and edges representing relationships between them.
   - Features for each node (stock) are loaded from corresponding CSV files.

2. **Sanitize Node Attributes:**
   - Ensure each node has the necessary attributes (`features`, `holders`, and `description`). Missing attributes are set to default values.

3. **Convert to PyTorch Geometric Format:**
   - The NetworkX graph is converted to a PyTorch Geometric data object, which is then used for training.

4. **Define and Train GraphSAGE Model:**
   - A GraphSAGE model is defined and trained over several epochs to learn node embeddings.
   - The embeddings are saved to a `.pt` file for later use.

### Code Example
```python
# Assuming you have followed the code structure in the provided scripts
# This script generates the embeddings and saves them as node_embeddings.pt

# Define GraphSAGE model, train, and save embeddings
model = GraphSAGEModel(input_dim, hidden_dim, output_dim, num_layers)
train_model(model, data_loader)
embeddings = extract_embeddings(model, data_loader)
torch.save(embeddings, 'output_pyg/node_embeddings.pt')
Output
The generated embeddings are saved in the output_pyg/node_embeddings.pt file.
Ticker Mapping
Overview
Ticker mapping associates each stock ticker with an index used in the node embeddings. This mapping is essential to translate between ticker names and their corresponding indices in the embedding space.

Steps
Create a CSV File:

Each line in the CSV file corresponds to a stock ticker and its index.
Ensure Consistency:

The number of tickers should match the number of embeddings.
Example
csv
Copy code
Ticker,Index
AAPL,0
MSFT,1
GOOGL,2
...
Output
The mapping is saved in ticker_mapping.csv.
Calculating Similarities
Overview
We compute the cosine similarity between all pairs of stock embeddings to determine how similar they are to each other.

Steps
Load Node Embeddings:

Load the embeddings from node_embeddings.pt.
Compute Cosine Similarity:

For each pair of stocks, compute the cosine similarity between their embedding vectors.
Filter Invalid Indices:

Ensure that only valid stock indices (less than 718) are considered in the similarity computation.
Save Results:

The computed similarities are saved to similarities.csv.
Code Example
python
Copy code
# Compute and save similarities
compute_and_save_similarities(embedding_file='output_pyg/node_embeddings.pt',
                              output_file='output_pyg/similarities.csv',
                              num_stocks=718)
Output
The similarities are saved in similarities.csv.
Extracting Top 3 Similar Stocks
Overview
After computing the similarities, the next step is to identify the top 3 most similar pairs of stocks.

Steps
Load Similarity Data:

Load the similarity scores from similarities.csv.
Identify Top 3 Pairs:

Sort the similarity scores and extract the top 3 pairs.
Map Indices to Tickers:

Convert the stock indices back to their corresponding ticker names using the ticker mapping.
Code Example
python
Copy code
# Extract top 3 most similar stocks
top_similarities = extract_top_3_similarities('output_pyg/similarities.csv',
                                              ticker_mapping='ticker_mapping.csv')
Output
The top 3 similar stock pairs are printed and can be saved to a file if needed.
Conclusion
By following these steps, you can generate node embeddings for stocks, calculate similarities between them, and identify the most similar stock pairs. This process is useful for various applications, including recommendation systems, clustering, and market analysis.

css
Copy code

Save this content as a `README.md` file in your project directory. It will provide a clear and detailed explanation of the entire process, making it easy to understand and reproduce the results.