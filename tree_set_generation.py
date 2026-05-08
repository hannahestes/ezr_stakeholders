#!/usr/bin/env python3
"""
Run EZR multiple times on datasets and collect results in JSON format with stability analysis.

Usage:
    python tree_set_generation.py --folder /path/to/data --runs 50

Outputs:
    EZR_trees/<datasetname>/
        ├── trees.json              (all trees with Option A stability: min feature frequency)
        └── features_summary.json   (Option B: per-feature stability across all runs)
"""

import os
import sys
import subprocess
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any

def check_and_install_requirements():
    """Check if ezr is installed; install if not."""
    try:
        import ezr
        print("[✓] ezr is already installed")
        return True
    except ImportError:
        print("[!] ezr not found. Installing from git...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "git+https://github.com/timm/ezr.git"],
                check=True,
                capture_output=True
            )
            print("[✓] ezr installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[✗] Failed to install ezr: {e.stderr.decode()}")
            return False

def parse_ezr_output(raw_output: str) -> Dict[str, Any]:
    """Parse ezr output to extract accuracy, features, and tree structure."""
    result = {
        "accuracy": None,
        "features": [],
        "tree_complexity": 0.0
    }
    
    # Extract hold-out accuracy (Best train: X hold-out: Y)
    accuracy_match = re.search(r"hold-out:\s*(-?\d+)", raw_output)
    if accuracy_match:
        result["accuracy"] = int(accuracy_match.group(1))
    
    # Extract features from "Used: feature1 feature2 feature3"
    features_match = re.search(r"Used:\s*(.*?)(?:\n|$)", raw_output)
    if features_match:
        features_str = features_match.group(1).strip()
        if features_str:
            result["features"] = features_str.split()
    
    # Estimate tree complexity based on feature count and tree depth
    lines = raw_output.split("\n")
    tree_lines = [l for l in lines if re.match(r"^n:\s+\d+", l)]
    
    if tree_lines:
        max_indent = max(len(re.match(r"^(\|*\s*)", l).group(1)) for l in tree_lines)
        depth_measure = max_indent / 20
        num_features = len(result["features"]) if result["features"] else 1
        feature_complexity = num_features / 10.0
        result["tree_complexity"] = round(min(1.0, (feature_complexity + depth_measure) / 2), 2)
    
    return result

def run_ezr_on_file(csv_path: str, run_num: int, base_seed: int) -> Dict[str, Any]:
    """Run ezr on a single CSV file and return parsed output."""
    try:
        seed = base_seed + run_num
        result = subprocess.run(
            ["ezr", "-f", str(csv_path), "-s", str(seed)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        raw_output = result.stdout
        if result.returncode != 0:
            raw_output = result.stderr or "EZR execution failed"
        
        parsed = parse_ezr_output(raw_output)
        
        return {
            "run": run_num,
            "accuracy": parsed["accuracy"],
            "stability": 0.0,
            "tree_complexity": parsed["tree_complexity"],
            "features": parsed["features"],
            "raw_output": raw_output
        }
    
    except subprocess.TimeoutExpired:
        print(f"  [!] Timeout running ezr on {csv_path}")
        return None
    except Exception as e:
        print(f"  [!] Error running ezr: {e}")
        return None

def process_dataset(csv_path: str, num_runs: int, base_seed: int) -> List[Dict[str, Any]]:
    """Run ezr multiple times on a dataset and collect results."""
    dataset_name = Path(csv_path).stem
    print(f"\nProcessing {dataset_name}...")
    
    trees = []
    for run_num in range(1, num_runs + 1):
        print(f"  Run {run_num}/{num_runs}...", end=" ", flush=True)
        tree_result = run_ezr_on_file(csv_path, run_num, base_seed)
        
        if tree_result:
            trees.append(tree_result)
            if run_num % 10 == 0 or run_num == num_runs:
              print(f"  Run {run_num}/{num_runs}")
        else:
            print("(failed)")
    
    return trees

def calculate_stability_option_a(trees: List[Dict], num_runs: int) -> List[Dict]:
    """
    Option A: Per-tree stability based on minimum feature frequency.
    
    A tree is only as stable as its least-stable feature.
    Stability = min(feature_frequencies) / num_runs * 100
    """
    if num_runs < 2:
        for tree in trees:
            tree["stability"] = 0.0
        return trees
    
    # Count feature frequencies across all runs
    all_features = set()
    for tree in trees:
        all_features.update(tree["features"])
    
    if not all_features:
        for tree in trees:
            tree["stability"] = 0.0
        return trees
    
    # Count how many times each feature appears
    feature_counts = {feat: 0 for feat in all_features}
    for tree in trees:
        for feat in tree["features"]:
            feature_counts[feat] += 1
    
    # For each tree: stability = min frequency / total runs * 100
    for tree in trees:
        if tree["features"]:
            min_freq = min(feature_counts[f] for f in tree["features"])
            tree["stability"] = round((min_freq / num_runs) * 100, 1)
        else:
            tree["stability"] = 0.0
    
    return trees

def calculate_feature_stability_option_b(trees: List[Dict], num_runs: int) -> Dict[str, Any]:
    """
    Option B: Per-feature stability across all runs.
    
    For each feature, calculate what percentage of trees it appears in.
    Returns a summary dict with feature stats ranked by stability.
    """
    all_features = set()
    for tree in trees:
        all_features.update(tree["features"])
    
    if not all_features:
        return {"total_runs": num_runs, "total_unique_features": 0, "features": []}
    
    # Count frequency of each feature
    feature_stats = {}
    for feat in all_features:
        count = sum(1 for tree in trees if feat in tree["features"])
        feature_stats[feat] = {
            "count": count,
            "stability": round((count / num_runs) * 100, 1),
            "appears_in_runs": count,
            "total_runs": num_runs
        }
    
    # Sort by stability (descending)
    sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]["stability"], reverse=True)
    
    return {
        "total_runs": num_runs,
        "total_unique_features": len(all_features),
        "features": [
            {
                "name": feat,
                "stability": stats["stability"],
                "appears_in_runs": stats["appears_in_runs"],
                "total_runs": num_runs
            }
            for feat, stats in sorted_features
        ]
    }

def main():
    parser = argparse.ArgumentParser(
        description="Run EZR multiple times on datasets in a folder with stability analysis"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to folder containing CSV files"
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=True,
        help="Number of times to run EZR per file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base random seed (each run uses seed + run_number). Default: 1"
    )
    
    args = parser.parse_args()
    
    # Validate folder
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"[✗] Folder not found: {args.folder}")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"[✗] Not a directory: {args.folder}")
        sys.exit(1)
    
    # Check and install requirements
    if not check_and_install_requirements():
        sys.exit(1)
    
    # Create main output directory
    output_root = Path("EZR_trees")
    output_root.mkdir(exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(folder_path.glob("*.csv"))
    if not csv_files:
        print(f"[!] No CSV files found in {args.folder}")
        sys.exit(1)
    
    print(f"[√] Found {len(csv_files)} CSV file(s)")
    print(f"[√] Using base seed: {args.seed}")
    
    total_runs = 0
    all_trees = []
    dataset_summaries = {}
    
    # Process each CSV file
    for csv_file in csv_files:
        dataset_name = csv_file.stem
        trees = process_dataset(str(csv_file), args.runs, args.seed)
        
        if trees:
            # Calculate Option A stability (per-tree, min feature frequency)
            trees = calculate_stability_option_a(trees, args.runs)
            
            # Calculate Option B stability (per-feature)
            feature_summary = calculate_feature_stability_option_b(trees, args.runs)
            
            all_trees.extend(trees)
            total_runs += len(trees)
            dataset_summaries[dataset_name] = feature_summary
            
            # Create dataset-specific folder
            dataset_folder = output_root / dataset_name
            dataset_folder.mkdir(exist_ok=True)
            
            # Save trees.json
            trees_file = dataset_folder / "trees.json"
            trees_data = {
                "total": len(trees),
                "dataset": dataset_name,
                "stability_method": "Option A (minimum feature frequency)",
                "trees": trees
            }
            with open(trees_file, "w") as f:
                json.dump(trees_data, f, indent=2)
            print(f"  [✓] Saved: {trees_file}")
            
            # Save features_summary.json
            summary_file = dataset_folder / "features_summary.json"
            summary_data = {
                "dataset": dataset_name,
                "stability_method": "Option B (per-feature stability)",
                "analysis": feature_summary
            }
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)
            print(f"  [✓] Saved: {summary_file}")
    
    # Summary
    if all_trees:
        print(f"\n[✓] Complete! Summary:")
        print(f"    Total runs: {total_runs}")
        print(f"    Datasets: {len(csv_files)}")
        print(f"    Output structure:")
        print(f"    EZR_trees/")
        for csv_file in csv_files:
            print(f"    └── {csv_file.stem}/")
            print(f"        ├── trees.json")
            print(f"        └── features_summary.json")
    else:
        print("\n[!] No successful runs to save")
        sys.exit(1)

if __name__ == "__main__":
    main()
