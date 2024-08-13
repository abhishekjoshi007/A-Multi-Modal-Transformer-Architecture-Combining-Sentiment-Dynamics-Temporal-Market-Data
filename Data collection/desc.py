import os
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_with_selenium(url):
    try:
        # Set up the WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run Chrome in headless mode
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)

        # Wait for the page to load
        driver.implicitly_wait(10)

        # Get the page source and parse it
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find the specific section for description
        section_description = soup.find('section', {'data-testid': 'description'})
        if not section_description:
            print('Section for description not found')
            description = None
        else:
            description_tag = section_description.find('p')
            if description_tag:
                description = description_tag.text.strip()
            else:
                description = None

        # Close the WebDriver
        driver.quit()

        return description
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def main():
    try:
        # Load tickers from CSV
        csv_path = '/Users/abhishekjoshi/Documents/GitHub/personalized-portfolio-recommendation/csv/Technology data_1.csv'  # Update this path if necessary
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        # Ensure proper parsing of CSV with commas within data fields
        tickers_df = pd.read_csv(csv_path, quotechar='"')

        # Drop rows where the 'Ticker' column is empty or contains NaN
        tickers_df = tickers_df.dropna(subset=['Ticker'])

        # Ensure the 'Ticker' column is treated as a string
        tickers_df['Ticker'] = tickers_df['Ticker'].astype(str)

        tickers = tickers_df['Ticker'].tolist()

        # Create the base directory for storing data
        base_dir = 'Technology_data'
        os.makedirs(base_dir, exist_ok=True)

        for ticker in tickers:
            # Construct the URL
            url = f'https://finance.yahoo.com/quote/{ticker}/profile/'

            # Scrape the data
            print(f"Scraping data for {ticker} from {url}")
            description = scrape_with_selenium(url)
            if description is not None:
                # Create a directory for the ticker
                ticker_dir = os.path.join(base_dir, ticker)
                os.makedirs(ticker_dir, exist_ok=True)

                # Save the data to a JSON file
                data = {
                    'description': description
                }
                json_filename = os.path.join(ticker_dir, 'description.json')
                with open(json_filename, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                
                print(f"Description for {ticker} saved to {json_filename}")
            else:
                print(f"Failed to scrape description for {ticker}")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
