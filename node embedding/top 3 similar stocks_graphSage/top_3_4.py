import heapq
import csv
from tqdm import tqdm

# Define file paths
temp_similarity_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/output_pyg/similarities.csv'
output_top3_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/top3_similarities.csv'
ticker_mapping_file = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/node embedding/graphsage/ticker_mapping.csv'

# Load ticker mapping
ticker_mapping = {}
with open(ticker_mapping_file, mode='r') as infile:
    reader = csv.reader(infile)
    next(reader)  # Skip the header row
    for rows in reader:
        ticker, index = rows
        ticker_mapping[index] = ticker  # Store the mapping in reverse (index -> ticker)

# Priority queue to store top 3 similarities
top_similarities = []

# Open the file and process line by line
with open(temp_similarity_file, 'r') as f_in:
    reader = csv.reader(f_in)
    next(reader)  # Skip header
    
    # Initialize tqdm progress bar
    total_lines = sum(1 for line in open(temp_similarity_file)) - 1  # Total lines minus the header
    progress_bar = tqdm(total=total_lines, desc="Processing similarities")
    
    for row in reader:
        stock1_idx, stock2_idx, similarity = row
        similarity = float(similarity)
        
        # Map indices to tickers
        stock1_ticker = ticker_mapping.get(stock1_idx.replace('Index ', ''), f"Index {stock1_idx}")
        stock2_ticker = ticker_mapping.get(stock2_idx.replace('Index ', ''), f"Index {stock2_idx}")
        
        # Push the similarity into the heap
        heapq.heappush(top_similarities, (similarity, stock1_ticker, stock2_ticker))
        
        # Keep only the top 3
        if len(top_similarities) > 3:
            heapq.heappop(top_similarities)
        
        # Update progress bar
        progress_bar.update(1)
    
    progress_bar.close()

# Write the top 3 similarities to the output file
with open(output_top3_file, 'w') as f_out:
    f_out.write("Stock 1,Stock 2,Similarity Score\n")
    for similarity, stock1_ticker, stock2_ticker in sorted(top_similarities, reverse=True):
        f_out.write(f"{stock1_ticker},{stock2_ticker},{similarity}\n")

print(f"Top 3 similar stocks have been computed and saved to {output_top3_file}")
