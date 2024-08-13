import requests
import json

# Prepare the payload for the API request using the updated 'spotId' and 'uuid'
api_url = "https://api-2-0.spot.im/v1.0.0/conversation/read"
payload = json.dumps({
  "conversation_id": "sp_Rba9aFpG_finmb$27444752",  # Updated to match the desired format
  "count": 250,
  "offset": 0,
  "sort_by": "newest"  # Assuming you want to sort by the newest; adjust as needed
})

api_headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
  'Content-Type': 'application/json',
  'x-spot-id': "sp_Rba9aFpG",  # Spot ID as per your configuration
  'x-post-id': "finmb$27444752",  # Post ID updated to reflect the desired conversation
  # Include any other necessary headers as per the API documentation or your requirements
}

# Make the API request to fetch the conversation data
response = requests.post(api_url, headers=api_headers, data=payload)

# Parse the JSON response and print it
data = response.json()
print(json.dumps(data, indent=4))  # Print the response data formatted for readability
