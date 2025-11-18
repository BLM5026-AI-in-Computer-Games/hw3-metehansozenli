#!/usr/bin/env python3
"""
Task 3: TSP Solver Comparison (Nearest Neighbor vs Google OR-Tools)

Runs many random Euclidean TSP instances (Task 1 style), compares a fast
Nearest Neighbor heuristic with a stronger OR-Tools based solver, and
produces plots + a short markdown report.
"""

import argparse
import time
import random
from pathlib import Path
from typing import Any, List, Tuple
from dataclasses import dataclass, asdict

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# Unified color scheme for all visualizations
COLORS = {
    # Background and grid
    'background': '#f8f9fa',
    'grid': '#e9ecef',
    
    # Algorithm-specific colors
    'nn_primary': '#FF6B6B',      # Coral Red for NN 
    'nn_light': '#FFA07A',        # Light Salmon for NN
    'or_primary': '#4ECDC4',      # Teal/Cyan for OR-Tools
    'or_light': '#95E1D3',        # Light Teal for OR-Tools
    
    # Node colors
    'nodes': '#A8DADC',           # Soft Blue for regular nodes
    'start_node': '#FF6B6B',      # Coral for start node (matches NN)
    
    # Edge colors
    'all_edges': '#CED4DA',       # Light Gray for complete graph edges
    
    # Text
    'title': '#2C3E50',           # Dark Blue-Gray for titles
    'text': '#34495E'             # Slate Gray for text
}


@dataclass
class ExperimentResult:
    """Single experiment result data structure"""
    instance_id: int
    graph_type: str  # here always "random" 
    n_nodes: int
    seed: int
    
    # Nearest Neighbor results
    nn_tour_length: float
    nn_runtime_s: float
    nn_tour: List[int]
    
    # OR-Tools results
    or_tour_length: float
    or_runtime_s: float
    or_tour: List[int]
    
    # Comparison metrics
    quality_gap_percent: float  # (NN - OR) / OR * 100
    speedup_factor: float  # NN_time / OR_time
    
    def to_dict(self):
        """Dict without full tours (for compact CSV)."""
        d = asdict(self)
        d.pop('nn_tour')
        d.pop('or_tour')
        return d


class NearestNeighborSolver:
    """Greedy heuristic TSP solver"""
    
    def solve(self, G: Any, nodes: List[Any], start_node: Any, weight: str = "length") -> Tuple[List[Any], float]:
        """
        Solve TSP using Nearest Neighbor heuristic
        Returns: (tour, runtime_seconds)
        """
        start_time = time.perf_counter()
        
        unvisited = set(nodes)
        tour = [start_node]
        unvisited.remove(start_node)
        
        while unvisited:
            current = tour[-1]
            # Find nearest unvisited node (direct edge weight)
            nearest = min(unvisited, key=lambda node: G[current][node][weight])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        tour.append(start_node)  # Return to start
        runtime = time.perf_counter() - start_time
        
        return tour, runtime


class ORToolsSolver:
    """Google OR-Tools based TSP solver"""
    
    def __init__(self, time_limit_s: int = 10):
        self.time_limit_s = time_limit_s
    
    def solve(self, G: Any, nodes: List[Any], start_node: Any, weight: str = "length") -> Tuple[List[Any], float]:
        """
        Solve TSP using Google OR-Tools
        Returns: (tour, runtime_seconds)
        """
        start_time = time.perf_counter()
        
        # Build node index mappings
        n = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for idx, node in enumerate(nodes)}
        start_idx = node_to_idx[start_node]
        
        # Distance scale for integer precision
        scale = 1000.0
        
        # Create OR-Tools routing model
        manager = pywrapcp.RoutingIndexManager(n, 1, start_idx)
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback using direct edge weights
        def distance_callback(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            if i == j:
                return 0
            u = nodes[i]
            v = nodes[j]
            # Direct edge weight from graph
            length = G[u][v][weight]
            return int(length * scale)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(self.time_limit_s)
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # Extract tour
            index = routing.Start(0)
            tour_indices = []
            while not routing.IsEnd(index):
                tour_indices.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            tour_indices.append(tour_indices[0])  # Close loop
            
            tour = [idx_to_node[idx] for idx in tour_indices]
        else:
            # Fallback
            tour = [start_node, start_node]
        
        runtime = time.perf_counter() - start_time
        return tour, runtime


def calculate_tour_length(G: Any, tour: List[Any], weight: str = "length") -> float:
    """Sum of edge weights along a tour."""
    return float(
        sum(G[tour[i]][tour[i + 1]][weight] for i in range(len(tour) - 1))
    )


def create_random_euclidean_graph(n: int, seed: int, area=(0.0, 100.0)) -> Tuple[Any, List[int]]:
    """Create complete Euclidean graph on n random points."""
    random.seed(seed)
    np.random.seed(seed)
    
    G = nx.Graph()
    
    # Generate random points
    for i in range(n):
        x = random.uniform(*area)
        y = random.uniform(*area)
        G.add_node(i, x=x, y=y, pos=(x, y))
    
    # Complete graph with Euclidean distances
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = G.nodes[i]['x'], G.nodes[i]['y']
            xj, yj = G.nodes[j]['x'], G.nodes[j]['y']
            distance = float(np.hypot(xi - xj, yi - yj))
            G.add_edge(i, j, length=distance)
    
    return G, list(range(n))


def run_experiment(instance_id: int, n: int, seed: int) -> Tuple[ExperimentResult, Any, List]:
    """Run one NN vs OR-Tools comparison and return result + graph for plotting."""
    
    print(f"  Experiment {instance_id}: random graph, n={n}, seed={seed}")
    
    # Create random Euclidean graph
    G, nodes = create_random_euclidean_graph(n, seed)
    start_node = nodes[0]
    
    # Nearest Neighbor
    nn_solver = NearestNeighborSolver()
    nn_tour, nn_time = nn_solver.solve(G, nodes, start_node, weight="length")
    nn_length = calculate_tour_length(G, nn_tour, weight="length")
    
    # OR-Tools
    or_solver = ORToolsSolver(time_limit_s=10)
    or_tour, or_time = or_solver.solve(G, nodes, start_node, weight="length")
    or_length = calculate_tour_length(G, or_tour, weight="length")
    
    # Comparison metrics
    quality_gap = ((nn_length - or_length) / or_length) * 100 if or_length > 0 else 0.0
    speedup = (or_time / nn_time) if nn_time > 0 else 1.0
    
    # Create index mapping for simple IDs
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    nn_tour_idx = [node_to_idx[node] if node in node_to_idx else -1 for node in nn_tour]
    or_tour_idx = [node_to_idx[node] if node in node_to_idx else -1 for node in or_tour]
    
    result = ExperimentResult(
        instance_id=instance_id,
        graph_type="random",
        n_nodes=n,
        seed=seed,
        nn_tour_length=nn_length,
        nn_runtime_s=nn_time,
        nn_tour=nn_tour_idx,
        or_tour_length=or_length,
        or_runtime_s=or_time,
        or_tour=or_tour_idx,
        quality_gap_percent=quality_gap,
        speedup_factor=speedup
    )
    
    print(f"     ✓ NN: {nn_length:.2f} ({nn_time:.4f}s) | OR: {or_length:.2f} ({or_time:.4f}s) | Gap: {quality_gap:.1f}%")
    
    return result, G, nodes


def visualize_comparison_example(G: nx.Graph, nodes: List, nn_tour: List, or_tour: List, 
                                  nn_length: float, or_length: float, output_dir: Path):
    """
    Create Task 1 style side-by-side visualization comparing NN and OR-Tools solutions.
    Shows both tours on the same graph topology.
    """
    
    pos = {i: (G.nodes[i]['x'], G.nodes[i]['y']) for i in nodes}
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    # ============= NEAREST NEIGHBOR =============
    ax = axes[0]
    ax.set_facecolor(COLORS['background'])
    ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    
    all_edges = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, 
                          edge_color=COLORS['all_edges'], 
                          width=0.8, alpha=0.3, ax=ax)
    
    # NN tour kenarlarını çiz
    tour_edges = [(nn_tour[i], nn_tour[i+1]) for i in range(len(nn_tour)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=tour_edges, 
                          edge_color=COLORS['nn_primary'], 
                          width=3, alpha=0.8, ax=ax)
    
    # Noktaları çiz (Task 1 style)
    node_colors = [COLORS['start_node'] if i == nodes[0] else COLORS['nodes'] for i in nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                          node_color=node_colors, 
                          node_size=600, 
                          alpha=0.9, ax=ax,
                          edgecolors='white', linewidths=2)
    
    # Numara etiketleri (0, 1, 2, ...)
    labels = {nodes[i]: str(i) for i in range(len(nodes))}
    nx.draw_networkx_labels(G, pos, labels=labels, 
                           font_size=12, font_weight='bold', 
                           font_color='white', ax=ax)
    
    # Başlık
    ax.set_title(f'Nearest Neighbor\nTour Length: {nn_length:.2f} units', 
                fontsize=16, fontweight='bold', color=COLORS['title'], pad=20)
    
    # Eksen ayarları
    ax.set_aspect('equal')
    margin = 5
    ax.set_xlim(-margin, 105)
    ax.set_ylim(-margin, 105)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['start_node'], 
                  markersize=12, label='Start Point (0)', markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['nodes'], 
                  markersize=12, label='Edges', markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], color=COLORS['nn_primary'], linewidth=3, label='NN Tour'),
        plt.Line2D([0], [0], color=COLORS['all_edges'], linewidth=1, alpha=0.3, label='All Connections')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # ============= OR-TOOLS =============
    ax = axes[1]
    ax.set_facecolor(COLORS['background'])
    ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tüm kenarları çiz (arka planda)
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, 
                          edge_color=COLORS['all_edges'], 
                          width=0.8, alpha=0.3, ax=ax)
    
    # OR-Tools tour kenarlarını çiz (yeşil)
    or_tour_edges = [(or_tour[i], or_tour[i+1]) for i in range(len(or_tour)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=or_tour_edges, 
                          edge_color=COLORS['or_primary'], 
                          width=3, alpha=0.8, ax=ax)
    
    # Noktaları çiz (Task 1 style)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                          node_color=node_colors, 
                          node_size=600, 
                          alpha=0.9, ax=ax,
                          edgecolors='white', linewidths=2)
    
    # Numara etiketleri
    nx.draw_networkx_labels(G, pos, labels=labels, 
                           font_size=12, font_weight='bold', 
                           font_color='white', ax=ax)
    
    # Başlık (quality gap ile)
    quality_gap = ((nn_length - or_length) / or_length) * 100
    ax.set_title(f'OR-Tools\nTour Length: {or_length:.2f} units (Gap: {quality_gap:.1f}%)', 
                fontsize=16, fontweight='bold', color=COLORS['title'], pad=20)
    
    # Eksen ayarları
    ax.set_aspect('equal')
    ax.set_xlim(-margin, 105)
    ax.set_ylim(-margin, 105)
    
    # Legend
    legend_elements_or = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['start_node'], 
                  markersize=12, label='Start Point (0)', markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['nodes'], 
                  markersize=12, label='Edges', markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], color=COLORS['or_primary'], linewidth=3, label='OR-Tools Tour'),
        plt.Line2D([0], [0], color=COLORS['all_edges'], linewidth=1, alpha=0.3, label='All Connections')
    ]
    ax.legend(handles=legend_elements_or, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_example_tours_visualization.png', dpi=200, 
               bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    
    print("  Saved: 5_example_tours_visualization.png")


def create_visualizations(results: List[ExperimentResult], output_dir: Path):
    """Generate core solution quality visualizations"""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Set style
    sns.set_style("whitegrid")
    
    # ==== 1. SOLUTION QUALITY COMPARISON (2 panels) ====
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot - Tour Length Distribution
    ax = axes[0]
    data = [df['nn_tour_length'], df['or_tour_length']]
    bp = ax.boxplot(data, labels=['Nearest Neighbor', 'OR-Tools'], patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['nn_light'])
    bp['boxes'][1].set_facecolor(COLORS['or_light'])
    ax.set_title('Tour Length Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Tour Length (units)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Scatter plot - Direct Comparison
    ax = axes[1]
    ax.scatter(df['nn_tour_length'], df['or_tour_length'], alpha=0.6, s=100, c=COLORS['nn_primary'], edgecolors='black', linewidth=0.5)
    min_val = min(df['nn_tour_length'].min(), df['or_tour_length'].min())
    max_val = max(df['nn_tour_length'].max(), df['or_tour_length'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal Performance Line')
    ax.set_xlabel('Nearest Neighbor Tour Length (units)', fontsize=12)
    ax.set_ylabel('OR-Tools Tour Length (units)', fontsize=12)
    ax.set_title('Solution Quality: NN vs OR-Tools', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_solution_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Saved: 1_solution_quality_comparison.png")
    
    # ==== 2. RUNTIME vs QUALITY TRADE-OFF ====
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot NN points
    ax.scatter(df['nn_runtime_s'], df['nn_tour_length'], 
              alpha=0.7, s=120, c=COLORS['nn_light'], marker='o', label='Nearest Neighbor', edgecolors='black', linewidth=1)
    
    # Plot OR-Tools points
    ax.scatter(df['or_runtime_s'], df['or_tour_length'], 
              alpha=0.7, s=120, c=COLORS['or_light'], marker='s', label='OR-Tools', edgecolors='black', linewidth=1)
    
    # Add average markers
    ax.scatter(df['nn_runtime_s'].mean(), df['nn_tour_length'].mean(),
              s=400, c=COLORS['nn_primary'], marker='*', label='NN Average', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(df['or_runtime_s'].mean(), df['or_tour_length'].mean(),
              s=400, c=COLORS['or_primary'], marker='*', label='OR-Tools Average', edgecolors='black', linewidth=2, zorder=5)
    
    ax.set_xlabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tour Length (units)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime vs Tour Length', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_runtime_quality_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Saved: 2_runtime_quality_tradeoff.png")

    # ==== 3. INSTANCE-BY-INSTANCE COMPARISON ====
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(results))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['nn_tour_length'], width, 
                   label='Nearest Neighbor', color=COLORS['nn_light'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df['or_tour_length'], width, 
                   label='OR-Tools', color=COLORS['or_light'], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Instance ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tour Length (units)', fontsize=12, fontweight='bold')
    ax.set_title('Instance-by-Instance Tour Length Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x[::max(1, len(results)//20)])  # Show every nth label to avoid clutter
    ax.set_xticklabels(df['instance_id'].values[::max(1, len(results)//20)])
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / '3_instance_by_instance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_instance_by_instance.png")
    
    # ==== 4. QUALITY GAP ANALYSIS ====
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Unified color scheme - all bars same color like other graphs
    bars = ax.bar(df['instance_id'], df['quality_gap_percent'], 
                  color=COLORS['nn_light'], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.axhline(y=df['quality_gap_percent'].mean(), color=COLORS['or_primary'], linestyle='--', 
              linewidth=2.5, label=f'Average Gap: {df["quality_gap_percent"].mean():.1f}%')
    
    ax.set_xlabel('Instance ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Quality Gap (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quality Gap Analysis by Instance', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, framealpha=0.95, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_quality_gap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 4_quality_gap_analysis.png")


def generate_report(results: List[ExperimentResult], output_path: Path):
    """Generate markdown report"""
    
    df = pd.DataFrame([r.to_dict() for r in results])  # in-memory only, no CSV export
    
    report = []
    report.append("# Task 3: TSP Solver Comparison Report\n")
    report.append("## Experimental Setup\n")
    report.append(f"- **Total Instances**: {len(results)}")
    report.append(f"- **Graph Types**: {', '.join(df['graph_type'].unique())}")
    report.append(f"- **Node Counts**: {df['n_nodes'].min()} - {df['n_nodes'].max()}")
    report.append(f"- **Algorithms**: Nearest Neighbor (Heuristic) vs OR-Tools (Advanced)\n")
    
    report.append("## Summary Statistics\n")
    report.append("### Tour Length")
    report.append(f"- **NN Average**: {df['nn_tour_length'].mean():.2f} ± {df['nn_tour_length'].std():.2f}")
    report.append(f"- **OR-Tools Average**: {df['or_tour_length'].mean():.2f} ± {df['or_tour_length'].std():.2f}")
    report.append(f"- **Average Quality Gap**: {df['quality_gap_percent'].mean():.2f}%\n")
    
    report.append("### Runtime")
    report.append(f"- **NN Average**: {df['nn_runtime_s'].mean():.4f}s ± {df['nn_runtime_s'].std():.4f}s")
    report.append(f"- **OR-Tools Average**: {df['or_runtime_s'].mean():.4f}s ± {df['or_runtime_s'].std():.4f}s")
    report.append(f"- **Average Speedup (NN/OR)**: {df['speedup_factor'].mean():.2f}x\n")
    
    # Statistical significance tests
    from scipy import stats
    
    report.append("## Statistical Analysis\n")
    
    # Paired t-test for tour lengths
    t_stat, p_value = stats.ttest_rel(df['nn_tour_length'], df['or_tour_length'])
    report.append("### Paired t-test (Tour Length)")
    report.append(f"- **t-statistic**: {t_stat:.4f}")
    report.append(f"- **p-value**: {p_value:.6f}")
    if p_value < 0.001:
        report.append(f"- **Result**: Highly significant (p < 0.001) - OR-Tools produces significantly shorter tours")
    elif p_value < 0.05:
        report.append(f"- **Result**: Significant (p < 0.05) - OR-Tools produces significantly shorter tours")
    else:
        report.append(f"- **Result**: Not significant (p ≥ 0.05)\n")
    
    # Effect size (Cohen's d)
    mean_diff = df['nn_tour_length'].mean() - df['or_tour_length'].mean()
    pooled_std = np.sqrt((df['nn_tour_length'].std()**2 + df['or_tour_length'].std()**2) / 2)
    cohens_d = mean_diff / pooled_std
    report.append(f"\n### Effect Size (Cohen's d)")
    report.append(f"- **Cohen's d**: {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    report.append(f"- **Interpretation**: {effect_interpretation.capitalize()} effect size")
    report.append(f"- **Practical meaning**: OR-Tools reduces tour length by {mean_diff:.2f} units on average ({df['quality_gap_percent'].mean():.1f}%)\n")
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_pvalue = stats.wilcoxon(df['nn_tour_length'], df['or_tour_length'])
    report.append("### Wilcoxon Signed-Rank Test (Non-parametric)")
    report.append(f"- **W-statistic**: {w_stat:.2f}")
    report.append(f"- **p-value**: {w_pvalue:.6f}")
    report.append(f"- **Result**: Confirms {'significant' if w_pvalue < 0.05 else 'non-significant'} difference\n")
    
    report.append("## Key Findings\n")
    or_wins = (df['quality_gap_percent'] > 0).sum()
    nn_wins = (df['quality_gap_percent'] < 0).sum()
    ties = (df['quality_gap_percent'] == 0).sum()
    
    report.append(f"- OR-Tools produced better solutions in **{or_wins}/{len(results)}** ({or_wins/len(results)*100:.1f}%) cases")
    report.append(f"- NN produced better solutions in **{nn_wins}/{len(results)}** ({nn_wins/len(results)*100:.1f}%) cases")
    report.append(f"- Ties: **{ties}/{len(results)}** ({ties/len(results)*100:.1f}%)")
    report.append(f"- NN is **{df['speedup_factor'].mean():.1f}x faster** on average\n")
    
    report.append("## Conclusion\n")
    report.append("OR-Tools provides consistently better solution quality at the cost of increased runtime. ")
    report.append("Nearest Neighbor is significantly faster but produces suboptimal solutions. ")
    report.append("The choice depends on the application: use NN for speed-critical scenarios, OR-Tools for quality-critical scenarios.\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"  Saved report: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Task 3: Comprehensive TSP Solver Comparison")
    parser.add_argument("--n", type=int, default=20, help="Number of nodes per instance")
    parser.add_argument("--instances", type=int, default=30, help="Number of test instances (≥30 recommended)")
    parser.add_argument("--seed-start", type=int, default=100, help="Starting seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nTask 3: TSP Solver Comparison - Comprehensive Analysis")
    print("=" * 70)
    print("Configuration:")
    print(f"  Nodes per instance : {args.n}")
    print(f"  Number of instances: {args.instances}")
    print(f"  Graph type         : Random Euclidean")
    print(f"  Starting seed      : {args.seed_start}")
    print(f"  Output directory   : {output_dir}")
    print()
    
    # Run experiments on random Euclidean graphs
    print("Running experiments on random graphs...")
    results = []
    example_data = None  # Store one example for visualization
    
    for i in range(args.instances):
        instance_id = i + 1
        seed = args.seed_start + i
        try:
            result, G, nodes = run_experiment(instance_id, args.n, seed)
            results.append(result)
            
            # Save an instance for example visualization
            if i == 15:
                example_data = (G, nodes, result)
                
        except Exception as e:
            print(f"     ❌ Failed: {e}")
    
    if not results:
        print("\n❌ No successful experiments!")
        return
    
    print(f"\nCompleted {len(results)} experiments.")
    
    # Convert results to DataFrame for analysis/plots
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results, output_dir)
        
    # Generate example comparison visualization
    if example_data:
        print("\nGenerating example comparison visualization...")
        G, nodes, result = example_data
        # Convert index-based tours back to actual node IDs
        nn_tour = [nodes[i] for i in result.nn_tour]
        or_tour = [nodes[i] for i in result.or_tour]
        visualize_comparison_example(
            G,
            nodes,
            nn_tour,
            or_tour,
            result.nn_tour_length,
            result.or_tour_length,
            output_dir,
        )
    
    # Generate report
    print("\nGenerating report...")
    generate_report(results, output_dir / "REPORT.md")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total Instances: {len(results)}")
    print(f"\nTour Length:")
    print(f"  NN:       {df['nn_tour_length'].mean():.2f} ± {df['nn_tour_length'].std():.2f}")
    print(f"  OR-Tools: {df['or_tour_length'].mean():.2f} ± {df['or_tour_length'].std():.2f}")
    print(f"\nRuntime:")
    print(f"  NN:       {df['nn_runtime_s'].mean():.4f}s ± {df['nn_runtime_s'].std():.4f}s")
    print(f"  OR-Tools: {df['or_runtime_s'].mean():.4f}s ± {df['or_runtime_s'].std():.4f}s")
    print(f"\nQuality Gap: {df['quality_gap_percent'].mean():.2f}% ± {df['quality_gap_percent'].std():.2f}%")
    print(f"Speedup Factor: {df['speedup_factor'].mean():.2f}x")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nVisualizations:")
    print("  1. Solution Quality Comparison      (1_solution_quality_comparison.png)")
    print("     - Box plot: Tour length distribution")
    print("     - Scatter: NN vs OR-Tools direct comparison")
    print("\n  2. Runtime-Quality Trade-off        (2_runtime_quality_tradeoff.png)")
    print("     - Shows computational cost vs solution quality")
    print("\n  3. Instance-by-Instance Analysis    (3_instance_by_instance.png)")
    print("     - Bar chart comparing each test case")
    print("\n  4. Quality Gap Analysis             (4_quality_gap_analysis.png)")
    print("     - Color-coded difficulty assessment")
    print("\n  5. Example Tours Visualization      (5_example_tours_visualization.png)")
    print("     - Side-by-side tour comparison (Task 1 style)")
    print("\nReport:")
    print("  - REPORT.md (includes statistical tests: t-test, Cohen's d, Wilcoxon)")
    print("\nTask 3 finished successfully.\n")


if __name__ == "__main__":
    main()
