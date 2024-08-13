import os
import requests
import json
import time
import pandas as pd

api_url = "https://api-2-0.spot.im/v1.0.0/conversation/read"
csv_path = 'Complete-List-of-SP-500-Index-Constituents-Apr-3-2024_1.csv'  # Update to the path of your CSV file
base_dir = 'historic_data'

def fetch_comments(payload, headers):
    comments = []
    comment_counter = 0
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            data = response.json()
            while data.get("conversation", {}).get("has_next", False):
                comm = data["conversation"]["comments"]
                comments.extend(comm)
                comment_counter += len(comm)
                print(f"Fetched {comment_counter} comments so far for conversation {payload['conversation_id']}...")

                payload["offset"] = data["conversation"]["offset"]
                time.sleep(1)  # Pause to avoid rate limits
                response = requests.post(api_url, headers=headers, data=json.dumps(payload))

                if response.status_code != 200:
                    print(f"Failed to fetch data: Status code {response.status_code}")
                    break
                data = response.json()
        else:
            print(f"Failed to fetch data: Status code {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")

    return comments

def main():
    data = pd.read_csv(csv_path)
    total_tickers = len(data)

    # Create main directory for historical data
    os.makedirs(base_dir, exist_ok=True)
    
    for idx, row in data.iterrows():
        payload = {
            "conversation_id": row['Conversation Id'],  # Ensure this matches your CSV header
            "count": 25,
            "offset": 0,
            "sort_by": "newest",
        }
        api_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0",
            "Content-Type": "application/json",
            "x-spot-id": row['X-Spot-Id'],  # Ensure this matches your CSV header
            "x-post-id": row['X-Post-Id'],  # Ensure this matches your CSV header
        }
        
        print(f"Processing {idx + 1}/{total_tickers}: {row['Ticker']} ({row['Company Name']})")
        comments = fetch_comments(payload, api_headers)
        
        # Create a directory for each ticker
        ticker_dir = os.path.join(base_dir, row['Ticker'])
        os.makedirs(ticker_dir, exist_ok=True)
        
        filename = os.path.join(ticker_dir, f"{row['Ticker']}_comments.json")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(comments, f, ensure_ascii=False, indent=4)
            print(f"Saved comments for {row['Ticker']} to {filename}")
        except Exception as e:
            print(f"Failed to write comments to file: {e}")

if __name__ == "__main__":
    main()
