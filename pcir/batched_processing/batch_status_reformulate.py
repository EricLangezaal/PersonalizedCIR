import json
import argparse
import os
import re
import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10), reraise=True)
def process_content(content, prompt_type):
    """
    Processes the content to extract 'rewrite' and 'response' based on prompt_type.
    """
    try:
        if prompt_type == 1:
            match_rewrite = re.search(r'Queries:\s*(.*)', content, re.DOTALL)
            match_response = re.search(r'Response:(.*?)Queries:', content, re.DOTALL)
        elif prompt_type == 2:
            match_rewrite = re.search(r'Rewrite:\s*(.*?)Response:', content, re.DOTALL)
            match_response = re.search(r'Response:\s*(.*)', content, re.DOTALL)

        if not (match_rewrite and match_response):
            raise ValueError("Failed to match required outputs in response.")

        return match_rewrite.group(1).strip(), match_response.group(1).strip()

    except Exception as e:
        print(f"Error processing content: {e}")
        raise

def load_batch_id_mapping(folder):
    """
    Load the appropriate batch_id_mapping file based on the folder.
    """
    if "2024" in folder:
        mapping_file = os.path.join(folder, "batch_id_mapping_2024.json")
    else:
        mapping_file = os.path.join(folder, "batch_id_mapping_2023.json")

    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"No mapping file found: {mapping_file}")
    with open(mapping_file, "r", encoding="utf-8") as f:
        return json.load(f)

def check_batch_status_for_combinations(folder, shot, annotation, prompt_type):
    try:
        batch_id_mapping = load_batch_id_mapping(folder)
    except FileNotFoundError as e:
        print(e)
        return

    # File pattern by annotation, shot, and prompt_type
    file_pattern = f"{annotation}_{shot}shot_prompt{prompt_type}"
    matching_batches = {
        filename: batch_id
        for filename, batch_id in batch_id_mapping.items()
        if file_pattern in filename
    }

    if not matching_batches:
        print(f"No batches found for: {shot} shot, {annotation} annotation, and prompt type {prompt_type}.")
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
                    processed_output_path=processed_path,
                    prompt_type=prompt_type,
                )
            elif batch.status == "in_progress":
                print(f"Batch for {filename} is still in progress.")
            else:
                print(f"Batch for {filename} has status: {batch.status}.")
        except Exception as e:
            print(f"Error for batch {filename}: {e}")

def download_and_process_results(file_id, processed_output_path, prompt_type):
    if os.path.exists(processed_output_path):
        print(f"File '{processed_output_path}' already exists. Skipping download and processing.")
        return

    result_file = client.files.content(file_id)
    try:
        with open(processed_output_path, "w", encoding="utf-8") as outfile:
            for line in result_file.text.strip().split("\n"):
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

                    rewrite, response = process_content(content, prompt_type)

                    processed_data = {
                        "sample_id": custom_id,
                        "rewrite_utt_text": rewrite,
                        "response_utt_text": response,
                    }

                    json.dump(processed_data, outfile)
                    outfile.write("\n")

                    print(f"Processed sample_id: {custom_id} --> SUCCESS")

                except Exception as e:
                    print(f"Error processing sample_id: {custom_id} - {e}")
                    continue

        print(f"Processed results saved to: {processed_output_path}")

    except Exception as e:
        print(f"Failed to write processed results to {processed_output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Check batch status and download & process results if ready.")
    parser.add_argument("--model_engine", type=str, default="gpt-4o-mini", help="GPT model engine.")
    parser.add_argument("--shot", type=int, default=0, help="Number of examples.")
    parser.add_argument("--annotation", type=str, default="human", help="Annotation type (None, All, human, automatic, or LLM).")
    parser.add_argument("--prompt_type", type=int, choices=[1, 2], default=1, help="Prompt type.")
    parser.add_argument("--year", type=int, choices=[2023, 2024], default=2023, help="Year of test topics.")
    args = parser.parse_args()

    folder = os.path.join("data", "batch", args.model_engine, str(args.year))
    check_batch_status_for_combinations(
        folder=folder,
        shot=args.shot,
        annotation=args.annotation,
        prompt_type=args.prompt_type,
    )

if __name__ == "__main__":
    main()
