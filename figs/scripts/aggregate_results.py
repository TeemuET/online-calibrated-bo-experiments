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

                # --- Create a single summary row for the entire run ---
                
                regret_history = metrics.get("y_regret_test", [])
                if not regret_history:
                    continue # Skip if there's no regret data

                result_row = {
                    # Key parameters from params.json
                    "problem_idx": params.get("problem_idx"),
                    "seed": params.get("seed"),
                    "dim": params.get("d"),
                    "surrogate": params.get("surrogate"),
                    "acquisition": params.get("acquisition"),
                    "recalibrate": params.get("recalibrate"),
                    
                    # --- Final Summary Metrics ---
                    
                    # 1. Best simple regret: how close did we get to the optimum? (Lower is better)
                    "best_simple_regret": min(regret_history),
                    
                    # 2. Cumulative regret: what was the total cost of exploration? (Sum of instantaneous regrets)
                    "final_cumulative_regret": sum(regret_history),
                    
                    # 3. Final values for other key metrics
                    "final_calibration_mse": metrics.get("y_calibration_mse", [None])[-1],
                    "final_sharpness": metrics.get("uct_sharpness", [None])[-1],
                    "final_nmse": metrics.get("nmse", [None])[-1],
                    "final_elpd": metrics.get("elpd", [None])[-1],
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
