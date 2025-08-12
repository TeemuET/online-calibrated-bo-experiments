import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

def aggregate_results(base_path):
    """
    Walks through the results directory, parses key metrics and parameters
    for each iteration, and returns a single pandas DataFrame.
    """
    all_results = []
    
    print(f"Searching for results in: {base_path}")

    # Use os.walk to find all parameter files
    for root, dirs, files in tqdm(os.walk(base_path), desc="Scanning Folders"):
        if "metrics.json" in files and "parameters.json" in files:
            try:
                # Load parameters for this run
                with open(os.path.join(root, "parameters.json")) as f:
                    params = json.load(f)
                
                # Load all metrics for this run
                with open(os.path.join(root, "metrics.json")) as f:
                    metrics = json.load(f)

                # The number of BO iterations is the length of a metric list
                num_iterations = len(metrics.get("y_regret_test", []))

                for i in range(num_iterations):
                    result_row = {
                        # Key parameters from params.json
                        "problem_idx": params.get("problem_idx"),
                        "seed": params.get("seed"),
                        "dim": params.get("d"),
                        "surrogate": params.get("surrogate"),
                        "acquisition": params.get("acquisition"),
                        "recalibrate": params.get("recalibrate"),
                        
                        # Iteration number
                        "bo_iteration": i,
                        
                        # --- Key Metrics from metrics.json ---
                        
                        # 1. Optimization Performance
                        "regret_test": metrics.get("y_regret_test", [None]*num_iterations)[i],
                        
                        # 2. Uncertainty Quality
                        "calibration_mse": metrics.get("y_calibration_mse", [None]*num_iterations)[i],
                        "sharpness": metrics.get("uct_sharpness", [None]*num_iterations)[i],
                        
                        # 3. Model Fit
                        "nmse": metrics.get("nmse", [None]*num_iterations)[i],
                        "elpd": metrics.get("elpd", [None]*num_iterations)[i],
                    }
                    all_results.append(result_row)

            except Exception as e:
                print(f"Could not process folder {root}. Error: {e}")

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    # Setup argument parser to be flexible
    parser = argparse.ArgumentParser(description="Aggregate experiment results into a single CSV.")
    parser.add_argument("path", type=str, help="Base path to the results directory (e.g., './results_synth_data/GP-EI-BENCHMARKS-notFixed').")
    args = parser.parse_args()
    
    df = aggregate_results(args.path)
    
    if not df.empty:
        output_file = "aggregated_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully aggregated {len(df.index)} data points into {output_file}")
        print("\nColumns created:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
    else:
        print("No results found to aggregate.")
