# Task 3: TSP Solver Comparison Report

## Experimental Setup

- **Total Instances**: 30
- **Graph Types**: random
- **Node Counts**: 15 - 15
- **Algorithms**: Nearest Neighbor (Heuristic) vs OR-Tools (Advanced)

## Summary Statistics

### Tour Length
- **NN Average**: 365.36 ± 41.43
- **OR-Tools Average**: 328.51 ± 31.38
- **Average Quality Gap**: 11.26%

### Runtime
- **NN Average**: 0.0001s ± 0.0000s
- **OR-Tools Average**: 10.0021s ± 0.0040s
- **Average Speedup (NN/OR)**: 73150.62x

## Statistical Analysis

### Paired t-test (Tour Length)
- **t-statistic**: 8.2448
- **p-value**: 0.000000
- **Result**: Highly significant (p < 0.001) - OR-Tools produces significantly shorter tours

### Effect Size (Cohen's d)
- **Cohen's d**: 1.0029
- **Interpretation**: Large effect size
- **Practical meaning**: OR-Tools reduces tour length by 36.85 units on average (11.3%)

### Wilcoxon Signed-Rank Test (Non-parametric)
- **W-statistic**: 0.00
- **p-value**: 0.000003
- **Result**: Confirms significant difference

## Key Findings

- OR-Tools produced better solutions in **29/30** (96.7%) cases
- NN produced better solutions in **0/30** (0.0%) cases
- Ties: **1/30** (3.3%)
- NN is **73150.6x faster** on average

## Conclusion

OR-Tools provides consistently better solution quality at the cost of increased runtime. 
Nearest Neighbor is significantly faster but produces suboptimal solutions. 
The choice depends on the application: use NN for speed-critical scenarios, OR-Tools for quality-critical scenarios.
