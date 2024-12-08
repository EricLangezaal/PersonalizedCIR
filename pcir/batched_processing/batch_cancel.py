import json
import requests
import os

# Set your OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Path to the JSON file containing batch IDs
BATCH_MAPPING_FILE = "data/batch/gpt-4o-mini/2024/batch_id_mapping_SAR_2024.json"

# OpenAI API endpoint to cancel a batch
CANCEL_URL = "https://api.openai.com/v1/batches/{batch_id}/cancel"

def cancel_batch(batch_id):
    """
    Cancel a batch using the OpenAI API.
    """
    url = CANCEL_URL.format(batch_id=batch_id)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        print(f"Successfully cancelled batch: {batch_id}")
    else:
        print(f"Failed to cancel batch: {batch_id}. Response: {response.text}")

def main():
    # Load batch IDs from the JSON file
    try:
        with open(BATCH_MAPPING_FILE, "r") as file:
            batch_mapping = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {BATCH_MAPPING_FILE}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return

    # Cancel each batch in the mapping
    for file_name, batch_id in batch_mapping.items():
        print(f"Attempting to cancel batch: {file_name} -> {batch_id}")
        cancel_batch(batch_id)

if __name__ == "__main__":
    main()