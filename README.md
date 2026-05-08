# EZR Stakeholder Decision Support System

A comprehensive pipeline for generating, analyzing, and explaining decision trees for multi-objective software engineering problems using COCOMO, EZR, and MOOT datasets.

## Overview

This toolkit supports research on **multi-stakeholder decision-making in software engineering**, specifically examining how different personas (Software Engineers, Product Managers) evaluate and prioritize decision trees across multiple objectives (accuracy, complexity, stability).

### Key Features

- ✅ **Run EZR N times** with deterministic seeding for diverse tree generation
- ✅ **Find Pareto frontier** across three optimization objectives
- ✅ **Generate 3D visualizations** of Pareto-optimal solutions
- ✅ **Create stakeholder-ready descriptions** with full tree details
- ✅ **Extract stability metrics** via Option A (minimum feature frequency) and Option B (per-feature stability)
- ✅ **Support for 10 COCOMO datasets** from MOOT

## Pipeline Overview

```
Input Datasets (COCOMO/MOOT CSVs)
           ↓
[1] run_ezr_batch.py       → Generate N trees per dataset
           ↓
Output: EZR_trees/
├── <dataset>/trees.json
└── features_summary.json
           ↓
[2] pareto_frontier.py      → Find Pareto-optimal trees
           ↓
Output: EZR_trees/
├── <dataset>/pareto_frontier.json
└── <dataset>/pareto_frontier.png (3D visualization)
           ↓
[3] generate_pareto_descriptions_final.py  → Create descriptions
           ↓
Output: EZR_trees/
├── <dataset>/pareto_descriptions.json
└── Ready for human studies
```

## Installation

### Requirements
- Python 3.11+
- EZR (installed automatically by script)
- Dependencies: `matplotlib`, `numpy`

### Setup

```bash
# Clone/download this repository
cd ezr_stakeholders

# No additional setup needed—EZR auto-installs
python run_ezr_batch.py --folder ./data --runs 10 --seed 42
```

## Usage

### Step 1: Generate Decision Trees

Run EZR N times on all CSV files in a folder:

```bash
python run_ezr_batch.py \
  --folder ./moot_data \
  --runs 500 \
  --seed 42
```

**Parameters:**
- `--folder`: Path to folder containing CSV files
- `--runs`: Number of times to run EZR per file (default: 500)
- `--seed`: Base random seed (each run uses seed + run_number, default: 42)

**Output:** `EZR_trees/` folder with per-dataset subfolders:
```
EZR_trees/
├── coc1000/
│   ├── trees.json
│   └── features_summary.json
├── xomo_osp2/
│   ├── trees.json
│   └── features_summary.json
└── ... (8 more datasets)
```

### Step 2: Find Pareto Frontier

Identify non-dominated trees across three objectives:

```bash
python pareto_frontier.py \
  --folder ./EZR_trees_500
```

**Objectives Optimized:**
- **Accuracy** (maximize) - Hold-out performance on test set
- **Stability** (maximize) - Feature frequency consistency across runs
- **Complexity** (minimize) - Tree depth & feature count combined

**Output:** Per-dataset frontier visualizations and JSON:
```
EZR_trees/
└── coc1000/
    ├── pareto_frontier.json    ← Frontier trees with metrics
    └── pareto_frontier.png     ← 3D scatter plot visualization
```

### Step 3: Generate Stakeholder Descriptions

Create detailed, human-readable summaries with full tree structure:

```bash
python generate_pareto_descriptions_final.py \
  --folder ./EZR_trees_500 \
  --verbose
```

**Output:** `pareto_descriptions.json` per dataset with formatted summaries:

```json
{
  "dataset": "coc1000",
  "num_trees": 8,
  "trees": [
    {
      "run": 106,
      "features": ["ACAP", "ARCH", "PCAP", "PCON", "PMAT", "PVOL"],
      "accuracy": 100,
      "stability": 22.8,
      "tree_complexity": 0.3,
      "num_nodes": 14,
      "train_value": 75,
      "holdout_value": 100,
      "formatted_summary": "..."
    }
  ]
}
```

## Key Concepts

### Stability (Option A)
**Definition:** Per-tree stability based on minimum feature frequency across N runs.

```
Stability = min(feature_frequencies) / num_runs * 100
```

A tree is only as stable as its least-stable feature. This conservative metric ensures all constituent features have demonstrated reliability.

### Stability (Option B)
**Definition:** Per-feature stability showing how often each attribute appears in the population of trees.

```
Feature Stability = count(trees containing feature) / total_trees * 100
```

Useful for identifying core stable features that consistently matter across the Rashomon set.

### Complexity
**Definition:** Normalized composite score combining tree depth and feature count.

```
Complexity = (Depth_norm + k × Attributes_norm) / (1 + k)

Where:
  Depth_norm = Actual_Depth / Max_Possible_Depth
  Attributes_norm = Num_Attributes / Max_Possible_Attributes
  k = 1.0 (equal weighting)
```

### Pareto Frontier
A tree is **non-dominated** (on the frontier) if no other tree is simultaneously:
- Higher accuracy
- Higher stability
- Lower complexity

The frontier represents the optimal trade-off space for stakeholders to choose from.

## Output Format: Formatted Summaries

Each frontier tree generates a detailed summary for stakeholder evaluation:

```
Tree Run 106
======================================================================

Features (6): ACAP, ARCH, PCAP, PCON, PMAT, PVOL

Performance:
  Hold-out Accuracy: 100
  Training Accuracy: 75
  Average node win: -2.1
  Win Range: [-59, 41]

Tree Structure:
  Complexity: 0.30
  Decision nodes: 14 total
    Good outcomes (>= 0): 7
    Bad outcomes (< 0): 7
  Max depth: 4

Root Decision: PVOL > 4.0

======================================================================
Raw EZR Output:
======================================================================

[Full EZR tree structure with all decision nodes...]
```

## Dataset Information

### COCOMO Datasets (from MOOT)

| Dataset | Description | Objectives |
|---------|-------------|-----------|
| COC1000 | COCOMO predecessor model | Reduce risk, effort, experience |
| NASA93dem | Real NASA project data | Similar to COCOMO |
| POM3 (A-D) | Agile project simulation | Idle rate, completion, cost |
| XOMO variants | JPL flight/ground systems | Effort, months, defects, risk |

**Y Objectives Example:**
```
Y: 5 LOC+ AEXP- PLEx- RISK- EFFORT-

Maximize:  Lines of Code (LOC+)
Minimize:  Analyst Experience (AEXP-)
Minimize:  Programming Language Experience (PLEx-)
Minimize:  Project Risk (RISK-)
Minimize:  Effort/Cost (EFFORT-)
```

## Research Context: ICSE 2026

This toolkit supports research on **multi-stakeholder decision-making in SE**, examining:

- **RQ1:** Do personas capture diversity of Software Engineer vs. Product Manager priorities?
- **RQ2:** Can role-tuned explanations help stakeholders make joint decisions?
- **RQ3:** What are limitations of personas as stakeholder proxies?

### Methodology Phases
1. **Phase 0:** Data preparation with Pareto frontier extraction
2. **Phase 1:** Generate descriptions & role-aware variants (using this toolkit)
3. **Phase 2:** Agent evaluation (6 personas × 2 roles)
4. **Phase 4:** Human validation study (N=20+ participants)
5. **Phase 5:** Consensus analysis & comparison

## Example Workflow

```bash
# 1. Generate 500 trees per dataset
python run_ezr_batch.py --folder ./moot_data --runs 500 --seed 42

# 2. Find Pareto frontiers (3 objectives: accuracy, stability, complexity)
python pareto_frontier.py --folder ./EZR_trees_500

# 3. Create stakeholder descriptions
python generate_pareto_descriptions_final.py --folder ./EZR_trees_500 --verbose

```

## File Structure

```
EZR_trees_500/
├── coc1000/
│   ├── trees.json                    # All 500 trees with raw_output
│   ├── features_summary.json         # Option B: per-feature stability
│   ├── pareto_frontier.json          # Frontier trees with metrics
│   ├── pareto_frontier.png           # 3D Pareto visualization
│   └── pareto_descriptions.json      # Formatted summaries (for study)
├── xomo_osp2/
│   └── ... (same structure)
└── ... (8 more datasets)
```

## License

This project is part of academic research. See LICENSE for details.

## Contact

For questions or issues, please open an issue or contact the research team.
