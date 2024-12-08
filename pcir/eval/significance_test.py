import argparse
import os
import json
import scipy.stats as stats


def main(files, agg_number, confidence, metrics_to_compare):

    all_data = []
    for f in files:
        with open(f, 'r') as file:
            json_data = json.load(file)
        name = list(json_data.keys())[0]

        all_data.append((name, json_data[name]))

    subsets = all_data[0][1].keys()
    for subset in subsets: 
        any_significant = False
        for i, (name1, data1) in enumerate(all_data):
            for name2, data2 in all_data[i+1:]:
                significant_metrics = []
                for metric in set(data1[subset].keys()) & set(metrics_to_compare):
                    mdata1 = data1[subset][metric]
                    mdata2 = data2[subset][metric]
                    pval = stats.ttest_ind_from_stats(
                        mdata1['mean'], mdata1['std'], agg_number,
                        mdata2['mean'], mdata2['std'], agg_number,
                    ).pvalue
                    if pval < confidence:
                        significant_metrics.append((metric, pval))

                if significant_metrics:
                    any_significant = True
                    metric_str = ', '.join([f"{metric} (p={pval:.4f})" for metric, pval in significant_metrics])
                    print(f"{name1} vs {name2} for {subset} is significant on {metric_str}")
        if any_significant:
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=None, help='Path to folder containing aggregated files.')
    parser.add_argument('--files', type=str, required=True, nargs='+', help='Paths to aggregated files')
    parser.add_argument('-n', '--number', default=5, type=int, help='Number of samples aggregated over.')
    parser.add_argument('-c', '--confidence', type=float, default=0.05, help='Upperbound for p-value.')
    parser.add_argument('-m', '--metrics', type=str, default=['MRR', 'NDCG@3', 'NDCG@5', 'MAP'], nargs='+')
    args = parser.parse_args()

    files = []
    for f in args.files:
        f = os.path.join(args.folder, f) if args.folder else f
        if os.path.isfile(f) and f.endswith('.json'):
            files.append(f)

    if len(files) < 2:
        raise argparse.ArgumentError(None, message='Need at least two valid .json files')
    
    print("Comparing the following files: ", files)
    print(f"Aggregating over {args.number} samples with two sided p-value upperbound {args.confidence} and metrics {args.metrics}\n\n")

    main(files, args.number, args.confidence, args.metrics)