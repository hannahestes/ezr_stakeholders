#!/usr/bin/env python3
"""
Generate natural language descriptions for Pareto frontier trees.

For each dataset, loads the pareto_frontier.json and generates 
structured summaries using the Tree DSL.

Usage:
    python generate_pareto_descriptions_final.py --folder ./EZR_trees_500
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ============================================================================
# TREE DSL
# ============================================================================

class TreeNode:
    """Base class for tree components."""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class Split(TreeNode):
    """A decision node that splits on a feature."""
    
    def __init__(self, feature: str, operator: str, threshold, samples: int, 
                 win: int, depth: int = 0):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.samples = samples
        self.win = win
        self.depth = depth
        self.left = None
        self.right = None


class Tree(TreeNode):
    """A complete decision tree with metadata."""
    
    def __init__(self, run_num: int, features: List[str], complexity: float, 
                 accuracy: int):
        self.run_num = run_num
        self.features = features
        self.complexity = complexity
        self.accuracy = accuracy
        self.root: Optional[Split] = None
        self.all_nodes: List[Split] = []


class TreeParser(TreeNode):
    """Parses EZR raw tree output into structured Tree objects."""
    
    def parse(self, raw_output: str, run_num: int, features: List[str], 
              complexity: float, accuracy: int) -> Tree:
        """Parse raw EZR output into a Tree."""
        tree = Tree(run_num, features, complexity, accuracy)
        
        lines = [l for l in raw_output.strip().split('\n') 
                 if l.strip() and not l.strip().startswith('#rows')]
        
        nodes_by_depth = defaultdict(list)
        
        for line in lines:
            if not line.strip():
                continue
            
            depth = line.count('|')
            parsed = self._parse_line(line, depth)
            
            if parsed:
                node = parsed
                nodes_by_depth[depth].append(node)
                tree.all_nodes.append(node)
        
        # Set root: prefer depth 0, fallback to first split node
        if nodes_by_depth[0]:
            tree.root = nodes_by_depth[0][0]
        elif tree.all_nodes:
            tree.root = tree.all_nodes[0]
        
        return tree
    
    def _parse_line(self, line: str, depth: int) -> Optional[Split]:
        """Parse a single tree line into a Split node."""
        clean = line.replace('|', '').strip()
        
        if not clean or 'if ' not in clean:
            return None
        
        # Check for "n:" prefix
        if not clean.startswith('n:'):
            return None
        
        # Remove "n:" prefix
        clean = clean[2:].strip()
        
        parts = clean.split()
        
        # Format is now: "5 win: 41 if PVOL > 4;"
        # We need: samples (5) and win value (41)
        if len(parts) < 4:
            return None
        
        try:
            samples = int(parts[0])
            # parts[1] should be 'win:'
            win = int(parts[2])
        except (ValueError, IndexError):
            return None
        
        # Extract and parse condition
        if 'if ' not in clean:
            return None
        
        if_idx = clean.index('if ')
        condition = clean[if_idx + 3:].strip().rstrip(';').strip()
        
        feature, operator, threshold = self._parse_condition(condition)
        
        if feature:
            return Split(
                feature=feature,
                operator=operator,
                threshold=threshold,
                samples=samples,
                win=win,
                depth=depth
            )
        
        return None
    
    def _parse_condition(self, condition: str) -> Tuple[Optional[str], Optional[str], any]:
        """Parse 'feature operator value' from condition string."""
        for op in ['==', '<=', '>=', '<', '>', '!=']:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    feature = parts[0].strip()
                    threshold = parts[1].strip()
                    
                    try:
                        threshold = float(threshold)
                    except ValueError:
                        pass
                    
                    return (feature, op, threshold)
        
        return (None, None, None)


class TreeAnalyzer(TreeNode):
    """Extract summary statistics from a tree."""
    
    def summarize(self, tree: Tree, raw_output: str = None) -> Dict:
        """Generate summary statistics for a tree."""
        
        if not tree.all_nodes:
            return self._empty_summary(tree, raw_output)
        
        # Get leaf nodes (nodes with best and worst outcomes)
        good_nodes, bad_nodes = self._split_nodes(tree, threshold=0)
        
        # Extract train/holdout from raw output
        train_val, holdout_val = self._extract_train_holdout(raw_output)
        
        summary = {
            'run': tree.run_num,
            'features': tree.features,
            'num_features': len(tree.features),
            'complexity': tree.complexity,
            'accuracy': tree.accuracy,
            'train_value': train_val,
            'holdout_value': holdout_val,
            'num_nodes': len(tree.all_nodes),
            'good_nodes': len(good_nodes),
            'bad_nodes': len(bad_nodes),
            'root_feature': tree.root.feature if tree.root else None,
            'root_operator': tree.root.operator if tree.root else None,
            'root_threshold': tree.root.threshold if tree.root else None,
            'avg_node_win': sum(n.win for n in tree.all_nodes) / len(tree.all_nodes) if tree.all_nodes else 0,
            'max_win': max((n.win for n in tree.all_nodes), default=0),
            'min_win': min((n.win for n in tree.all_nodes), default=0),
            'tree_depth': max((n.depth for n in tree.all_nodes), default=0) if tree.all_nodes else 0,
        }
        
        return summary
    
    def _empty_summary(self, tree: Tree, raw_output: str = None) -> Dict:
        """Return empty summary for unparseable tree."""
        train_val, holdout_val = self._extract_train_holdout(raw_output)
        
        return {
            'run': tree.run_num,
            'features': tree.features,
            'num_features': len(tree.features),
            'complexity': tree.complexity,
            'accuracy': tree.accuracy,
            'train_value': train_val,
            'holdout_value': holdout_val,
            'num_nodes': 0,
            'good_nodes': 0,
            'bad_nodes': 0,
            'root_feature': None,
            'root_operator': None,
            'root_threshold': None,
            'avg_node_win': 0,
            'max_win': 0,
            'min_win': 0,
            'tree_depth': 0,
        }
    
    def _extract_train_holdout(self, raw_output: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract 'Best train' and 'hold-out' values from raw output."""
        if not raw_output:
            return None, None
        
        match = re.search(r'Best train:\s*(\d+)\s+hold-out:\s*(\d+)', raw_output)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    
    def _split_nodes(self, tree: Tree, threshold: int) -> Tuple[List[Split], List[Split]]:
        """Split nodes into good (>= threshold) and bad (< threshold)."""
        good = [n for n in tree.all_nodes if n.win >= threshold]
        bad = [n for n in tree.all_nodes if n.win < threshold]
        return good, bad
    
    def format_summary(self, summary: Dict, raw_output: str = None) -> str:
        """Format summary as human-readable text with optional raw output."""
        
        root_split = f"{summary['root_feature']} {summary['root_operator']} {summary['root_threshold']}"
        
        text = f"""
Tree Run {summary['run']}
{'='*70}

Features ({summary['num_features']}): {', '.join(summary['features'])}

Performance:
  Hold-out Accuracy: {summary['holdout_value'] if summary['holdout_value'] is not None else summary['accuracy']}
  Training Accuracy: {summary['train_value'] if summary['train_value'] is not None else 'N/A'}
  Average node win: {summary['avg_node_win']:.1f}
  Win Range: [{summary['min_win']}, {summary['max_win']}]

Tree Structure:
  Complexity: {summary['complexity']:.2f}
  Decision nodes: {summary['num_nodes']} total
    Good outcomes (>= 0): {summary['good_nodes']}
    Bad outcomes (< 0): {summary['bad_nodes']}
  Max depth: {summary['tree_depth']}

Root Decision: {root_split}
"""
        
        if raw_output:
            text += f"\n{'='*70}\nRaw EZR Output:\n{'='*70}\n{raw_output}"
        
        return text.strip()


# ============================================================================
# DESCRIPTION GENERATOR
# ============================================================================

def generate_descriptions(dataset_folder: Path) -> Optional[List[Dict]]:
    """
    Generate descriptions for Pareto frontier trees.
    
    Returns list of tree descriptions with summaries.
    """
    # Load frontier
    frontier_file = dataset_folder / 'pareto_frontier.json'
    if not frontier_file.exists():
        return None
    
    with open(frontier_file, 'r') as f:
        frontier_data = json.load(f)
    
    frontier = frontier_data.get('frontier', [])
    
    # Load all trees to get raw_output
    trees_file = dataset_folder / 'trees.json'
    if not trees_file.exists():
        return None
    
    with open(trees_file, 'r') as f:
        trees_data = json.load(f)
    
    # Build lookup of run_num -> full tree data
    trees_lookup = {}
    for tree in trees_data.get('trees', []):
        run = tree.get('run', tree.get('run_num'))
        trees_lookup[run] = tree
    
    # Process frontier trees
    parser = TreeParser()
    analyzer = TreeAnalyzer()
    descriptions = []
    
    for frontier_tree in frontier:
        run_num = frontier_tree['run']
        
        # Get full tree data
        if run_num not in trees_lookup:
            continue
        
        full_tree_data = trees_lookup[run_num]
        
        # Parse tree
        parsed_tree = parser.parse(
            full_tree_data['raw_output'],
            run_num,
            frontier_tree['features'],
            frontier_tree['tree_complexity'],
            frontier_tree['accuracy']
        )
        
        # Analyze
        summary = analyzer.summarize(parsed_tree, raw_output=full_tree_data['raw_output'])
        
        # Add frontier metrics
        summary['stability'] = frontier_tree['stability']
        summary['formatted_summary'] = analyzer.format_summary(
            summary, 
            raw_output=full_tree_data['raw_output']
        )
        
        descriptions.append(summary)
    
    return descriptions


def save_descriptions(dataset_folder: Path, descriptions: List[Dict]):
    """Save descriptions to JSON file."""
    output_file = dataset_folder / 'pareto_descriptions.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': dataset_folder.name,
            'num_trees': len(descriptions),
            'trees': descriptions
        }, f, indent=2)
    
    return output_file


def print_descriptions(descriptions: List[Dict], dataset_name: str):
    """Print descriptions to console."""
    print(f"\n{'='*75}")
    print(f"Pareto Frontier Descriptions: {dataset_name}")
    print(f"{'='*75}")
    
    for desc in descriptions:
        print(desc['formatted_summary'])
        print(f"\nStability: {desc.get('stability', 'N/A')}%")
        print(f"\n{'-'*75}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate descriptions for Pareto frontier trees'
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to EZR_trees root folder'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print descriptions to console'
    )
    
    args = parser.parse_args()
    
    root_folder = Path(args.folder)
    
    if not root_folder.exists():
        print(f"[✗] Folder not found: {args.folder}")
        return
    
    # Find all dataset folders
    dataset_folders = sorted([
        d for d in root_folder.iterdir()
        if d.is_dir() and (d / 'pareto_frontier.json').exists()
    ])
    
    if not dataset_folders:
        print(f"[!] No Pareto frontiers found in {args.folder}")
        return
    
    print(f"\n[√] Found {len(dataset_folders)} dataset(s)")
    print("="*75)
    
    for dataset_folder in dataset_folders:
        dataset_name = dataset_folder.name
        print(f"\nProcessing {dataset_name}...")
        
        descriptions = generate_descriptions(dataset_folder)
        
        if not descriptions:
            print(f"  [!] Failed to generate descriptions")
            continue
        
        # Save
        output_file = save_descriptions(dataset_folder, descriptions)
        print(f"  [✓] Saved {len(descriptions)} descriptions to:")
        print(f"      {output_file}")
        
        # Print if verbose
        if args.verbose:
            print_descriptions(descriptions, dataset_name)
    
    print("\n" + "="*75)
    print(f"[✓] Complete! Descriptions saved to each dataset folder")


if __name__ == '__main__':
    main()