import pandas as pd
import glob
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
})

RESULTS_DIR = "/home/iailab73/khanm2/reglangs-cl/results/transformer_hyperparameter_search/cycle_navigation_small/"  

files = glob.glob(f"{RESULTS_DIR}/*.csv")

rows = []

pattern = re.compile(
    r"length_(?P<length>\d+)_"
    r"hidden_size_(?P<hidden_size>\d+)_"
    r"num_heads_(?P<num_heads>\d+)_"
    r"num_layers_(?P<num_layers>\d+)_"
    r"seed_(?P<seed>\d+)_"
    r"weight_init_(?P<weight_init>.+?)_"
    r"learning_rate_(?P<learning_rate>[\d\.e-]+)_"
    r"batch_size_(?P<batch_size>\d+)_"
    r"output_activation_(?P<output_activation>.+?)_"
    r"dropout_(?P<dropout>[\d\.]+)_"
    r"loss_function_(?P<loss_function>.+)"
)

for file in files:
    name = Path(file).stem
    match = pattern.search(name)
    if not match:
        continue

    params = match.groupdict()

    df = pd.read_csv(file)
    final_row = df.iloc[-1]

    params["final_loss"] = final_row["loss"]
    params["final_accuracy"] = final_row["accuracy"]

    rows.append(params)

print("Number of matched files:", len(rows))
print("Example filename:", Path(files[0]).stem if files else "No files found")

results_df = pd.DataFrame(rows)

results_df = results_df.astype({
    "length": int,
    "hidden_size": int,
    "num_heads": int,
    "num_layers": int,
    "batch_size": int,
    "seed": int,
    "learning_rate": float,
    "dropout": float,
    "final_loss": float,
    "final_accuracy": float,
})

print("Columns:", results_df.columns)

# Sort by best performance
results_df = results_df.sort_values(
    by=["final_accuracy", "final_loss"],
    ascending=[False, True]
)

print("Top 10 Results:")
print(results_df.head(10))

# Parameters to analyze (excluding batch_size)
params = [
    ("hidden_size", "Hidden Size"),
    ("num_heads", "Num Heads"),
    ("num_layers", "Num Layers"),
    ("dropout", "Dropout"),
]

# Create pairwise heatmaps
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

param_pairs = [
    (("hidden_size", "Hidden Size"), ("num_heads", "Num Heads")),
    (("hidden_size", "Hidden Size"), ("num_layers", "Num Layers")),
    (("hidden_size", "Hidden Size"), ("dropout", "Dropout")),
    (("num_heads", "Num Heads"), ("num_layers", "Num Layers")),
    (("num_heads", "Num Heads"), ("dropout", "Dropout")),
    (("num_layers", "Num Layers"), ("dropout", "Dropout")),
]

for idx, (param1, param2) in enumerate(param_pairs):
    ax = axes[idx]
    
    # Create pivot table with mean accuracy
    pivot = results_df.pivot_table(
        values='final_accuracy',
        index=param2[0],
        columns=param1[0],
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=pivot.mean().mean(),
        ax=ax,
        cbar_kws={'label': 'Mean Accuracy'}
    )
    ax.set_title(f'{param1[1]} vs {param2[1]}')
    ax.set_xlabel(param1[1])
    ax.set_ylabel(param2[1])

fig.suptitle("Hyperparameter Search Results - Pairwise Heatmaps", fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig("heatmaps_pairwise.png", dpi=300, bbox_inches='tight')
plt.show()

# Individual parameter analysis (bar plots)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (col, label) in zip(axes, params):
    means = results_df.groupby(col)["final_accuracy"].mean()
    stds = results_df.groupby(col)["final_accuracy"].std()
    
    x_pos = np.arange(len(means))
    ax.bar(x_pos, means.values, yerr=stds.values, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(means.index.astype(str))
    ax.set_xlabel(label)
    ax.set_ylabel("Mean Accuracy")
    ax.set_title(f"Mean Accuracy vs {label}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("individual_params.png", dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("\n=== Summary Statistics ===")
for col, label in params:
    print(f"\n{label}:")
    summary = results_df.groupby(col)["final_accuracy"].agg(['mean', 'std', 'count'])
    print(summary)

# Best configuration
print("\n=== Best Configuration ===")
best = results_df.iloc[0]
print(f"Accuracy: {best['final_accuracy']:.4f}")
print(f"Loss: {best['final_loss']:.4f}")
print(f"Hidden Size: {best['hidden_size']}")
print(f"Num Heads: {best['num_heads']}")
print(f"Num Layers: {best['num_layers']}")
print(f"Dropout: {best['dropout']}")
print(f"Learning Rate: {best['learning_rate']}")