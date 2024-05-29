import argparse
import json
import logging
import os

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_without_rounding(number, precision=3):
    # Truncate the number to the specified precision without rounding
    truncated_number = int(number * (10**precision)) / (10**precision)
    formatted_number = f"{truncated_number:.{precision}f}"
    return formatted_number


def extract_metrics(path):
    metrics = {
        "train": {"mse": [], "mae": [], "pcorr": []},
        "eval": {"mse": [], "mae": [], "pcorr": []}
    }
    
    # Recursively walk through the directory to find output.json files
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "output.json":
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for subset in ["train", "eval"]:
                        for metric in metrics[subset].keys():
                            metrics[subset][metric].append(data[subset][metric])
    
    # Calculate mean and std for each metric
    final_metrics = {}
    for subset in metrics:
        final_metrics[subset] = {}
        for metric, values in metrics[subset].items():
            final_metrics[subset][metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
    
    return final_metrics

def main(args):
    output_dir = args.output_dir
    metrics = extract_metrics(output_dir)
    metrics[f"train_report"] = " & ".join([f'${format_without_rounding(v["mean"], 3)} \pm {format_without_rounding(v["std"], 4)}$' for k, v in  metrics["train"].items()])
    metrics[f"eval_report"] = " & ".join([f'${format_without_rounding(v["mean"], 3)} \pm {format_without_rounding(v["std"], 4)}$' for k, v in  metrics["eval"].items()])

    avg_output_path = os.path.join(output_dir, "avg_output.json")
    with open(avg_output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--output_dir', type=str, default="", help='provide output directory')
    args = parser.parse_args()
    main(args)
