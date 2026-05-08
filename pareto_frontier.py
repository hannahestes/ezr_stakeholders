#!/usr/bin/env python3
"""
Pareto Frontier Analysis for Decision Trees

For each dataset folder, finds the Pareto frontier across accuracy, 
stability, and tree_complexity. Generates 3D visualization.

Usage:
    python pareto_frontier.py --folder ./EZR_trees
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def is_dominated(tree_a: Dict, tree_b: Dict) -> bool:
    """
    Check if tree_a is dominated by tree_b.
    
    Objectives:
    - accuracy: maximize
    - stability: maximize  
    - tree_complexity: minimize
    """
    acc_ge = tree_b['accuracy'] >= tree_a['accuracy']
    stab_ge = tree_b['stability'] >= tree_a['stability']
    cx_le = tree_b['tree_complexity'] <= tree_a['tree_complexity']
    
    if not (acc_ge and stab_ge and cx_le):
        return False
    
    acc_gt = tree_b['accuracy'] > tree_a['accuracy']
    stab_gt = tree_b['stability'] > tree_a['stability']
    cx_lt = tree_b['tree_complexity'] < tree_a['tree_complexity']
    
    return acc_gt or stab_gt or cx_lt


def find_pareto_frontier(trees: List[Dict]) -> List[Dict]:
    """Find Pareto frontier (non-dominated trees)."""
    frontier = []
    
    for candidate in trees:
        is_candidate_dominated = False
        
        for frontier_tree in frontier:
            if is_dominated(candidate, frontier_tree):
                is_candidate_dominated = True
                break
        
        if is_candidate_dominated:
            continue
        
        frontier = [t for t in frontier if not is_dominated(t, candidate)]
        frontier.append(candidate)
    
    return sorted(frontier, key=lambda t: t['accuracy'], reverse=True)


def load_trees_from_json(trees_file: Path) -> List[Dict]:
    """Load trees from trees.json file."""
    with open(trees_file, 'r') as f:
        data = json.load(f)
    
    trees_list = data.get('trees', [])
    
    # Add run number if not present
    for i, tree in enumerate(trees_list, 1):
        if 'run' not in tree:
            tree['run'] = tree.get('run_num', i)
    
    return trees_list


def create_visualization(frontier: List[Dict], dataset_name: str, 
                        output_path: Path) -> str:
    """
    Create 3D scatter plot of Pareto frontier.
    Returns path to saved figure.
    """
    if not frontier:
        print(f"  [!] No frontier to visualize for {dataset_name}")
        return None
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    accuracy = [t['accuracy'] for t in frontier]
    stability = [t['stability'] for t in frontier]
    complexity = [t['tree_complexity'] for t in frontier]
    runs = [t['run'] for t in frontier]
    
    # Color by accuracy (gradient from red to green)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(frontier)))
    
    # Plot frontier points
    scatter = ax.scatter(accuracy, stability, complexity, 
                        c=range(len(frontier)), cmap='RdYlGn',
                        s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Label each point with run number
    for i, (acc, stab, cx, run) in enumerate(zip(accuracy, stability, complexity, runs)):
        ax.text(acc, stab, cx, f'  R{run}', fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_ylabel('Stability (%)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Complexity', fontsize=11, fontweight='bold')
    ax.set_title(f'Pareto Frontier: {dataset_name}\n({len(frontier)} trees)', 
                fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Frontier Order', fontsize=10)
    
    # Grid and style
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    
    # Save figure
    fig_path = output_path / 'pareto_frontier.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(fig_path)


def print_frontier(frontier: List[Dict], dataset_name: str):
    """Pretty-print the Pareto frontier."""
    print(f"\n  Pareto Frontier for {dataset_name}: {len(frontier)} trees")
    print(f"  {'-'*70}")
    print(f"  {'Run':>4} {'Accuracy':>10} {'Stability':>10} {'Complexity':>10} {'Features':<20}")
    print(f"  {'-'*70}")
    
    for tree in frontier:
        features_str = ', '.join(tree['features'][:2])
        if len(tree['features']) > 2:
            features_str += '...'
        
        print(f"  {tree['run']:4d} {tree['accuracy']:10d} "
              f"{tree['stability']:10.1f} {tree['tree_complexity']:10.2f} "
              f"{features_str:<20}")
    
    print(f"  {'-'*70}")


def process_dataset(dataset_folder: Path) -> Tuple[List[Dict], int]:
    """
    Process a single dataset folder.
    Returns (frontier, num_trees).
    """
    trees_file = dataset_folder / 'trees.json'
    
    if not trees_file.exists():
        return None, 0
    
    # Load trees
    trees = load_trees_from_json(trees_file)
    
    if not trees:
        return None, 0
    
    # Find frontier
    frontier = find_pareto_frontier(trees)
    
    return frontier, len(trees)


def main():
    parser = argparse.ArgumentParser(
        description='Find Pareto frontier for decision tree results'
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to EZR_trees root folder'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    root_folder = Path(args.folder)
    
    if not root_folder.exists():
        print(f"[✗] Folder not found: {args.folder}")
        return
    
    if not root_folder.is_dir():
        print(f"[✗] Not a directory: {args.folder}")
        return
    
    # Find all dataset folders (those containing trees.json)
    dataset_folders = sorted([
        d for d in root_folder.iterdir()
        if d.is_dir() and (d / 'trees.json').exists()
    ])
    
    if not dataset_folders:
        print(f"[!] No dataset folders found in {args.folder}")
        return
    
    print(f"\n[√] Found {len(dataset_folders)} dataset(s)")
    print("="*75)
    
    # Process each dataset
    results = {}
    
    for dataset_folder in dataset_folders:
        dataset_name = dataset_folder.name
        print(f"\nProcessing {dataset_name}...")
        
        frontier, num_trees = process_dataset(dataset_folder)
        
        if frontier is None:
            print(f"  [!] Failed to load trees")
            continue
        
        results[dataset_name] = {
            'frontier': frontier,
            'num_trees': num_trees,
            'num_frontier': len(frontier)
        }
        
        # Print summary
        print_frontier(frontier, dataset_name)
        
        # Save frontier
        frontier_file = dataset_folder / 'pareto_frontier.json'
        with open(frontier_file, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'total_trees': num_trees,
                'frontier_size': len(frontier),
                'frontier': frontier
            }, f, indent=2)
        print(f"  [✓] Saved: {frontier_file}")
        
        # Create visualization
        if not args.no_viz:
            fig_path = create_visualization(frontier, dataset_name, dataset_folder)
            if fig_path:
                print(f"  [✓] Saved: {fig_path}")
    
    # Summary
    print("\n" + "="*75)
    print("SUMMARY")
    print("="*75)
    
    for dataset_name in sorted(results.keys()):
        r = results[dataset_name]
        print(f"{dataset_name:30s} | "
              f"Total: {r['num_trees']:3d} | "
              f"Frontier: {r['num_frontier']:3d} "
              f"({100*r['num_frontier']/r['num_trees']:.1f}%)")
    
    print("="*75)
    print(f"[✓] Complete! Results saved to {root_folder}")


if __name__ == '__main__':
    main()