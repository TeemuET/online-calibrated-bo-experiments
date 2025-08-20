import os
import json
import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np

def aggregate_results(base_path):
    """
    Walks through the results directory, parses key metrics and parameters
    for each iteration, and returns a single pandas DataFrame.
    """
    all_results = []
    
    print(f"Searching for results in: {base_path}")

    # Use os.walk to find all parameter files
    for root, dirs, files in tqdm(os.walk(base_path), desc="Scanning Folders"):
        if "metrics.json" in files and "parameters.json" in files and "dataset.json" in files:
            try:
                # Load parameters for this run
                with open(os.path.join(root, "parameters.json")) as f:
                    params = json.load(f)
                
                # Load all metrics for this run
                with open(os.path.join(root, "metrics.json")) as f:
                    metrics = json.load(f)

                # Load dataset summary to get the scaling factor
                with open(os.path.join(root, "dataset.json")) as f:
                    dataset_summary = json.load(f)
                
                y_std = dataset_summary.get("y_std", 1.0) # Default to 1.0 if not found

                # --- Create a single summary row for the entire run ---
                
                regret_history_test = metrics.get("y_regret_test", [])
                regret_history_pool = metrics.get("y_regret_pool", [])

                result_row = {
                    # Key parameters from params.json
                    "problem_idx": params.get("problem_idx"),
                    "seed": params.get("seed"),
                    "dim": params.get("d"),
                    "surrogate": params.get("surrogate"),
                    "acquisition": params.get("acquisition"),
                    "recalibrate": params.get("recalibrate"),
                    "y_std_original": y_std, # Save the scaling factor
                    
                    # --- Final Summary Metrics ---
                    
                    # 1. Test Set Regret: Best simple and final cumulative
                    "best_simple_regret_test": min(regret_history_test) if regret_history_test else None,
                    "best_simple_regret_test_rescaled": min(regret_history_test) * y_std if regret_history_test else None,
                    "final_cumulative_regret_test": sum(regret_history_test) if regret_history_test else None,
                    "final_cumulative_regret_test_rescaled": sum(regret_history_test) * y_std if regret_history_test else None,

                    # 2. Pool Regret: Best simple and final cumulative
                    "best_simple_regret_pool": min(regret_history_pool) if regret_history_pool else None,
                    "best_simple_regret_pool_rescaled": min(regret_history_pool) * y_std if regret_history_pool else None,
                    "final_cumulative_regret_pool": sum(regret_history_pool) if regret_history_pool else None,
                    "final_cumulative_regret_pool_rescaled": sum(regret_history_pool) * y_std if regret_history_pool else None,
                    
                    # 3. Mean values for other key metrics over the entire run
                    "mean_calibration_mse": np.mean(metrics.get("y_calibration_mse", [])) if metrics.get("y_calibration_mse") else None,
                    "mean_uct_sharpness": np.mean(metrics.get("uct_sharpness", [])) if metrics.get("uct_sharpness") else None,
                    "mean_gaussian_sharpness": np.mean(metrics.get("gaussian_sharpness", [])) if metrics.get("gaussian_sharpness") else None,
                    "mean_nmse": np.mean(metrics.get("nmse", [])) if metrics.get("nmse") else None,
                    "mean_elpd": np.mean(metrics.get("elpd", [])) if metrics.get("elpd") else None,
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
        # Get the experiment name from the last part of the input path.
        experiment_name = os.path.basename(os.path.normpath(args.path))
        
        # Create a dynamic output filename based on the experiment.
        output_file = f"aggregated_{experiment_name}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully aggregated {len(df.index)} runs into {output_file}")
        print("\nColumns created:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
    else:
        print("No results found to aggregate.")
