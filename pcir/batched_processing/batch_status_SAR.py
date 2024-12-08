import json
import argparse
import os
import re
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
def process_content(content):
    """
    Processes the content to extract 'Provenance', 'Rewrite', and 'Response'.
    """
    try:
        match_prov = re.search(r'Provenance:\s*(.*?)\nRewrite:', content, re.DOTALL)
        match_rewrite = re.search(r'Rewrite:\s*(.*?)\nResponse:', content, re.DOTALL)
        match_response = re.search(r'Response:\s*(.*)', content, re.DOTALL)

        if not (match_prov and match_rewrite and match_response):
            raise ValueError("Failed to match required outputs in response.")

        provenance = match_prov.group(1).strip()
        rewrite = match_rewrite.group(1).strip()
        response = match_response.group(1).strip()

        return provenance, rewrite, response

    except Exception as e:
        print(f"Error processing content: {e}")
        raise

def load_batch_id_mapping(folder):

    if "2024" in folder:
        mapping_file = os.path.join(folder, "batch_id_mapping_SAR_2024.json")
    else:
        mapping_file = os.path.join(folder, "batch_id_mapping_SAR_2023.json")

    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"No mapping file found: {mapping_file}")
    with open(mapping_file, "r", encoding="utf-8") as f:
        return json.load(f)

def check_batch_status_for_combinations(folder, shot):
    try:
        batch_id_mapping = load_batch_id_mapping(folder)
    except FileNotFoundError as e:
        print(e)
        return

    # File pattern by shot
    file_pattern = f"SAR_{shot}shot"
    matching_batches = {
        filename: batch_id for filename, batch_id in batch_id_mapping.items() if file_pattern in filename
    }

    if not matching_batches:
        print(f"No batches found for: SAR {shot} shot.")
        return

    for filename, batch_id in matching_batches.items():
        try:
            batch = client.batches.retrieve(batch_id)
            print(f"Batch '{filename}' status: {batch.status}")
            if batch.status == "completed":
                print(f"Batch for '{filename}' completed. Downloading + processing...")
                processed_folder = os.path.join(folder, "processed")
                os.makedirs(processed_folder, exist_ok=True)
                processed_path = os.path.join(processed_folder, filename)
                download_and_process_results(
                    file_id=batch.output_file_id,
                    processed_output_path=processed_path
                )
            elif batch.status == "in_progress":
                print(f"Batch for {filename} is still in progress.")
            else:
                print(f"Batch for {filename} has status: {batch.status}.")
        except Exception as e:
            print(f"Error for batch {filename}: {e}")

def download_and_process_results(file_id, processed_output_path):
    if os.path.exists(processed_output_path):
        print(f"File '{processed_output_path}' already exists. Skipping download and processing.")
        return

    result_file = client.files.content(file_id)
    try:
        with open(processed_output_path, "w", encoding="utf-8") as outfile:
            for line in result_file.text.strip().split('\n'):
                try:
                    batch_entry = json.loads(line)
                    custom_id = batch_entry.get("custom_id")
                    response_body = batch_entry.get("response", {}).get("body", {})
                    choices = response_body.get("choices", [])
                    if not choices:
                        print(f"No choices found for custom_id: {custom_id}")
                        continue
                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        print(f"No content found for custom_id: {custom_id}")
                        continue

                    provenance, rewrite, response = process_content(content)

                    processed_data = {
                        "sample_id": custom_id,
                        "LLM_select": provenance,
                        "rewrite_utt_text": rewrite,
                        "response_utt_text": response
                    }

                    json.dump(processed_data, outfile)
                    outfile.write('\n')

                    print(f"Processed sample_id: {custom_id} --> SUCCESS")

                except Exception as e:
                    print(f"Error processing sample_id: {custom_id} - {e}")
                    continue

        print(f"Processed results saved to: {processed_output_path}")

    except Exception as e:
        print(f"Failed to write processed results to {processed_output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Check batch status and download & process results for SAR.")
    parser.add_argument("--model_engine", type=str, default="gpt-4o-mini", help="GPT model engine.")
    parser.add_argument("--shot", type=int, choices=[0, 1, 3, 5], default=0, help="Number of few-shot examples.")
    parser.add_argument("--year", type=int, choices=[2023, 2024], default=2023, help="Year of test topics.")
    args = parser.parse_args()

    folder = os.path.join("data", "batch", args.model_engine, str(args.year))
    check_batch_status_for_combinations(
        folder=folder,
        shot=args.shot
    )

if __name__ == "__main__":
    main()
