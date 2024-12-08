import os
import json
import argparse
import logging
from collections import defaultdict
import numpy as np
from utils import calculate_trec_res_NDCG, get_run_files

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_metrics(metrics_list):
    aggregated = {}
    metric_keys = metrics_list[0].keys()
    for key in metric_keys:
        values = [m[key] for m in metrics_list]
        aggregated[key] = {
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "variance": np.var(values),
            "median": np.median(values)
        }
    return aggregated

def process_combination(run_files, qrel_file, rel_threshold, automatic_method, year):
    all_metrics = defaultdict(list)

    for run_file in run_files:
        print(f"Processing run file: {run_file}")
        # Full test
        metrics_full = calculate_trec_res_NDCG(
            run_file=run_file,
            qrel_file=qrel_file,
            rel_threshold=rel_threshold,
            automatic_method=automatic_method,
            subset=False,
            year=year
        )
        all_metrics['full_set'].append(metrics_full)

        # Original subset
        metrics_subset = calculate_trec_res_NDCG(
            run_file=run_file,
            qrel_file=qrel_file,
            rel_threshold=rel_threshold,
            automatic_method=automatic_method,
            subset=True,
            year=year
        )
        all_metrics['subset'].append(metrics_subset)

        # Corrected subset
        metrics_corrected_subset = calculate_trec_res_NDCG(
            run_file=run_file,
            qrel_file=qrel_file,
            rel_threshold=rel_threshold,
            automatic_method=automatic_method,
            subset=True,
            new_ptkb=True,
            year=year
        )
        all_metrics['corrected_subset'].append(metrics_corrected_subset)

    aggregated_results = {}
    for method, metrics in all_metrics.items():
        if automatic_method:
            baseline_metrics = [m[0] for m in metrics]
            always_ptkb_metrics = [m[1] for m in metrics]
            aggregated_results[f"{method}_baseline"] = aggregate_metrics(baseline_metrics)
            aggregated_results[f"{method}_always_ptkb"] = aggregate_metrics(always_ptkb_metrics)
        else:
            aggregated_results[method] = aggregate_metrics(metrics)

    return aggregated_results

def main(args):
    final_results = {}

    base_name = args.input
    suffix = args.suffix if args.suffix else ''
    logging.info(f"Processing input: {base_name} with suffix: {suffix}")

    # Adjust trec_directory based on the year
    if args.trec_directory is None:
        args.trec_directory = f'data/results/trec_files/{args.year}/'

    # Adjust qrel_file based on the year
    if args.qrel_file is None:
        args.qrel_file = f"data/{args.year}-qrels.all-turns.txt"

    run_files = get_run_files(base_name, args.trec_directory, suffix=suffix, num_runs=args.num_runs)
    if not run_files:
        print(f"No run files found for {base_name} in {args.trec_directory}")
        return

    # Set the output file path
    if args.output_file is None:
        output_dir = f'data/batch/gpt-4o-mini/{args.year}/aggregated/'
        os.makedirs(output_dir, exist_ok=True)
        args.output_file = os.path.join(output_dir, f'aggregated_{args.input}{suffix}.json')

    # Check if the aggregated results already exist
    if os.path.exists(args.output_file):
        logging.info(f"Aggregated results already exist at {args.output_file}, skipping.")
        return

    aggregated = process_combination(
        run_files=run_files,
        qrel_file=args.qrel_file,
        rel_threshold=args.rel_threshold,
        automatic_method=args.automatic_method,
        year=args.year
    )
    final_results[base_name] = aggregated

    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=4)

    logging.info(f"Aggregated results saved to {args.output_file}")

if __name__ == "__main__":
    print("EXECUTING")
    parser = argparse.ArgumentParser(description="Aggregate TREC run results")
    parser.add_argument("--input", required=True, help="Input base name to process, including '_run'.")
    parser.add_argument("--suffix", default='', help="Suffix to append after run number in filenames.")
    parser.add_argument("--year", type=str, choices=['2023', '2024'], required=True, help="Year of the TREC files.")
    parser.add_argument("--trec_directory", type=str, default=None, help="TREC run directory.")
    parser.add_argument("--output_file", type=str, help="Optional: Path to save the aggregated results (JSON format).")
    parser.add_argument("--qrel_file", type=str, default=None, help="Path to the qrel file.")
    parser.add_argument("--rel_threshold", type=int, default=1, help="Relevance threshold for evaluation.")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to aggregate")
    parser.add_argument("--automatic_method", action='store_true', help="Use automatic method.")

    args = parser.parse_args()
    main(args)