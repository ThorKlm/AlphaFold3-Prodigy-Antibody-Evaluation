"""
Visualization functions for AlphaFold3 binding evaluation.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


def plot_mse_distributions(df, output_path,
                          x_col="Combination",
                          y_col="Center_Position_MSE",
                          title="Binding Pose Consistency Across AlphaFold3 Seeds",
                          figsize=(12, 6)):
    """
    Create a strip plot of MSE values with error bars.

    Args:
        df: DataFrame containing MSE data
        output_path: Path to save the plot
        x_col: Column name for x-axis categories
        y_col: Column name for y-axis values
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)

    # Generate consistent color palette
    palette = sns.color_palette("tab10", n_colors=df[x_col].nunique())
    combo_to_color = dict(zip(df[x_col].unique(), palette))

    # Plot individual points
    sns.stripplot(data=df, x=x_col, y=y_col, jitter=True,
                  alpha=0.7, size=6, hue=x_col, palette=combo_to_color, legend=False)

    # Overlay group mean ± std
    grouped = df.groupby(x_col)[y_col]
    means = grouped.mean()
    stds = grouped.std()
    positions = range(len(means))

    for i, combo in enumerate(means.index):
        plt.errorbar(x=i, y=means[combo], yerr=stds[combo], fmt='o',
                   color=combo_to_color[combo], capsize=5)

    # Final styling
    plt.xticks(positions, means.index, rotation=45, ha='right')
    plt.ylabel("Center-Aligned MSE (Å²)")
    plt.xlabel("Binding Entity–Antigen Combination")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_binding_energy_boxplot(df: pd.DataFrame, output_path: Union[str, Path],
                               x_col: str = "Combination",
                               y_col: str = "Estimated_DeltaG_kcal_per_mol",
                               title: str = "Estimated Binding Energy Across Seeds",
                               figsize: tuple = (10, 6)) -> None:
    """
    Create a boxplot of binding energy values.

    Args:
        df: DataFrame containing binding energy data
        output_path: Path to save the plot
        x_col: Column name for x-axis categories
        y_col: Column name for y-axis values
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)

    # Generate consistent color palette
    palette = sns.color_palette("tab10", n_colors=df[x_col].nunique())
    combo_to_color = dict(zip(df[x_col].unique(), palette))

    # Create boxplot
    sns.boxplot(data=df, x=x_col, y=y_col, hue=x_col, palette=combo_to_color, legend=False)

    # Add individual points
    sns.stripplot(data=df, x=x_col, y=y_col, color='black',
                jitter=True, alpha=0.6)

    # Final styling
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Estimated ΔG (kcal/mol)")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_energy_vs_mse(df: pd.DataFrame, output_path: Union[str, Path],
                      x_col: str = "Interface_Deviation_RMSD",
                      y_col: str = "Mean_Estimated_DeltaG_kcal_per_mol",
                      combo_col: str = "Combination",
                      title: str = "Mean ΔG vs Mean Pose Deviation per Combination",
                      figsize: tuple = (8, 6),
                      show_regression: bool = True) -> None:
    """
    Create a scatter plot of mean binding energy vs mean pose deviation.

    Args:
        df: DataFrame containing summary data
        output_path: Path to save the plot
        x_col: Column name for x-axis (pose deviation)
        y_col: Column name for y-axis (binding energy)
        combo_col: Column name for combination labels
        title: Plot title
        figsize: Figure size as (width, height) tuple
        show_regression: Whether to show regression line
    """
    plt.figure(figsize=figsize)

    # Generate consistent color palette
    palette = sns.color_palette("tab10", n_colors=df[combo_col].nunique())
    combo_to_color = dict(zip(df[combo_col].unique(), palette))

    # Create scatter plot
    for _, row in df.iterrows():
        if not pd.isna(row[y_col]):
            plt.scatter(row[x_col], row[y_col],
                      color=combo_to_color[row[combo_col]],
                      s=100, label=row[combo_col])

    # Add trendline if requested
    if show_regression and len(df) >= 3:
        try:
            # Filter out NaN values
            valid_data = df[[x_col, y_col]].dropna()
            if len(valid_data) >= 3:
                sns.regplot(data=valid_data, x=x_col, y=y_col,
                          scatter=False, ci=None, color='grey',
                          line_kws={'linestyle': '--', 'alpha': 0.7})
        except:
            pass  # Skip trendline if it fails

    # Add labels for each point
    for _, row in df.iterrows():
        if not pd.isna(row[y_col]):
            plt.annotate(
                f"{row['Binding_Entity']}-{row['Antigen']}",
                (row[x_col], row[y_col]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )

    # Final styling
    plt.xlabel("Mean Seed Pose Deviation (Å)")
    plt.ylabel("Mean Estimated ΔG (kcal/mol)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)

    # Add legend if not too many combinations
    if df[combo_col].nunique() <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_confusion_matrix_heatmap(df, output_path,
                                    binding_col="Binding_Entity",
                                    antigen_col="Antigen",
                                    value_col="Mean_Estimated_DeltaG_kcal_per_mol",
                                    title="Binding Energy Confusion Matrix",
                                    figsize=(14, 12),
                                    cmap="YlOrRd"):
    """
    Create a heatmap of binding energies for all binding entity-antigen combinations.

    Args:
        df: DataFrame containing binding energy data
        output_path: Path to save the plot
        binding_col: Column name for binding entities
        antigen_col: Column name for antigens
        value_col: Column name for binding energy values
        title: Plot title
        figsize: Figure size as (width, height) tuple
        cmap: Colormap for the heatmap
    """
    # Sort the names alphabetically (case insensitive)
    binding_entities = sorted(df[binding_col].unique(), key=str.lower)
    antigens = sorted(df[antigen_col].unique(), key=str.lower)

    # Remove duplicates and pivot
    df_unique = df.drop_duplicates(subset=[binding_col, antigen_col], keep='first')
    matrix = df_unique.pivot(index=binding_col, columns=antigen_col, values=value_col)
    matrix = matrix.reindex(index=binding_entities, columns=antigens)

    # Make the plot
    plt.figure(figsize=figsize)

    # For MSE, use viridis colormap with reverse (lower is better)
    if 'MSE' in value_col or 'mse' in value_col.lower():
        cmap = 'viridis_r'
        colorbar_label = 'Antigen Aligned Center Position MSE (Å²)'
    else:
        # For binding energy, use YlOrRd (more negative is better)
        cmap = 'YlOrRd'
        colorbar_label = value_col.replace('_', ' ')

    # Create heatmap with exact styling from ColabFold
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',  # 3 decimal places
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': colorbar_label},
        annot_kws={'size': 18}  # font size for annotations
    )

    # Make colorbar label bigger
    cbar = ax.collections[0].colorbar
    cbar.set_label(colorbar_label, size=20)

    # Highlight the diagonal - these should be the real binding pairs
    for i in range(min(len(binding_entities), len(antigens))):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=5))

    # Title and labels with exact font sizes
    plt.title(title, fontsize=22, pad=20)
    plt.xlabel('Antigens', fontsize=22)
    plt.ylabel('Binding Entities', fontsize=22)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_combined_plot(df_energy: pd.DataFrame, df_mse: pd.DataFrame, output_path: Union[str, Path],
                        binding_col: str = "Binding_Entity",
                        antigen_col: str = "Antigen",
                        combo_col: str = "Combination",
                        energy_col: str = "Mean_Estimated_DeltaG_kcal_per_mol",
                        mse_col: str = "Center_Position_MSE",
                        title: str = "AlphaFold3 Binding Evaluation",
                        figsize: tuple = (12, 10)) -> None:
    """
    Create a combined visualization with binding energy and MSE.

    Args:
        df_energy: DataFrame containing binding energy data
        df_mse: DataFrame containing MSE data
        output_path: Path to save the plot
        binding_col: Column name for binding entities
        antigen_col: Column name for antigens
        combo_col: Column name for combinations
        energy_col: Column name for binding energy values
        mse_col: Column name for MSE values
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    # Merge dataframes
    df_combined = pd.merge(df_energy, df_mse, on=[binding_col, antigen_col], suffixes=('', '_mse'))

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})

    # Plot 1: Binding energy heatmap
    pivot_energy = df_combined.pivot_table(index=binding_col, columns=antigen_col, values=energy_col)
    sns.heatmap(pivot_energy, annot=True, fmt=".2f", cmap="coolwarm_r", ax=ax1,
              cbar_kws={'label': 'Mean Estimated ΔG (kcal/mol)'})
    ax1.set_title("Binding Energy Confusion Matrix")

    # Plot 2: MSE heatmap
    pivot_mse = df_combined.pivot_table(index=binding_col, columns=antigen_col, values=mse_col)
    sns.heatmap(pivot_mse, annot=True, fmt=".2f", cmap="viridis_r", ax=ax2,
              cbar_kws={'label': 'Center-Aligned MSE (Å²)'})
    ax2.set_title("Binding Pose Consistency Across Seeds")

    # Final styling
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_matrix_visualization(df_energy: pd.DataFrame, df_mse: pd.DataFrame,
                              output_path: Union[str, Path],
                              scale_two_chain: bool = False,
                              title: str = "AlphaFold3 Binding Evaluation Matrix",
                              figsize: tuple = (16, 12)) -> None:
    """
    Create a comprehensive matrix visualization showing binding energy and MSE.

    Args:
        df_energy: DataFrame with binding energy summary
        df_mse: DataFrame with MSE results
        output_path: Path to save the plot
        scale_two_chain: Whether binding energies are scaled
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    # Merge dataframes
    df_combined = pd.merge(
        df_energy,
        df_mse[["Combination", "Center_Position_MSE"]],
        on="Combination",
        how="left"
    )

    # Get unique binding entities and antigens
    binding_entities = sorted(df_combined["Binding_Entity"].unique())
    antigens = sorted(df_combined["Antigen"].unique())

    # Create matrices for binding energies and MSE
    approx_energy_matrix = pd.pivot_table(
        df_combined,
        values="Mean_Approx_DeltaG_kcal_per_mol",
        index="Binding_Entity",
        columns="Antigen",
        fill_value=np.nan
    ).reindex(index=binding_entities, columns=antigens)

    prodigy_energy_matrix = pd.pivot_table(
        df_combined,
        values="Mean_Prodigy_DeltaG_kcal_per_mol",
        index="Binding_Entity",
        columns="Antigen",
        fill_value=np.nan
    ).reindex(index=binding_entities, columns=antigens)

    mse_matrix = pd.pivot_table(
        df_combined,
        values="Center_Position_MSE",
        index="Binding_Entity",
        columns="Antigen",
        fill_value=np.nan
    ).reindex(index=binding_entities, columns=antigens)

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, width_ratios=[10, 10, 1], height_ratios=[1, 10, 10])

    # Add title
    plt.suptitle(title, fontsize=16, y=0.98)

    # Create custom colormaps
    approx_energy_cmap = LinearSegmentedColormap.from_list(
        "approx_energy_cmap", ["#ffcccc", "#ff0000", "#990000", "#550000"]
    )
    prodigy_energy_cmap = LinearSegmentedColormap.from_list(
        "prodigy_energy_cmap", ["#ffddcc", "#ff7700", "#993300", "#552200"]
    )
    mse_cmap = LinearSegmentedColormap.from_list(
        "mse_cmap", ["#ccffcc", "#00ff00", "#009900", "#005500"]
    )

    # Plot antibody type indicators
    ax_types = fig.add_subplot(gs[0, 0:2])
    ax_types.axis("off")

    # Create a bar on top showing single chain vs two chain
    type_colors = {1: "#e6f2ff", 2: "#ffe6e6"}  # Blue for single chain, Red for two chain
    for i, entity in enumerate(binding_entities):
        chain_type = df_combined[df_combined["Binding_Entity"] == entity]["Antibody_Type"].iloc[0]
        rect = plt.Rectangle(
            (i, 0),
            1,
            1,
            color=type_colors[chain_type]
        )
        ax_types.add_patch(rect)
        ax_types.text(
            i + 0.5,
            0.5,
            f"{chain_type}-chain",
            ha="center",
            va="center",
            fontsize=10
        )

    ax_types.set_xlim(0, len(binding_entities))
    ax_types.set_ylim(0, 1)
    ax_types.set_title("Binding Entity Type", fontsize=14)

    # Plot Approximate Binding Energy heatmap
    ax_approx = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        approx_energy_matrix,
        cmap=approx_energy_cmap,
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax_approx,
        linewidths=0.5,
        center=0
    )
    energy_label = "Approximate ΔG (kcal/mol)"
    if scale_two_chain:
        energy_label += " (Scaled)"
    ax_approx.set_title(energy_label, fontsize=14)
    ax_approx.set_xlabel("Antigen", fontsize=12)
    ax_approx.set_ylabel("Binding Entity", fontsize=12)

    # Plot PRODIGY Binding Energy heatmap
    ax_prodigy = fig.add_subplot(gs[2, 0])
    sns.heatmap(
        prodigy_energy_matrix,
        cmap=prodigy_energy_cmap,
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax_prodigy,
        linewidths=0.5,
        center=0
    )
    prodigy_label = "PRODIGY ΔG (kcal/mol)"
    if scale_two_chain:
        prodigy_label += " (Scaled)"
    ax_prodigy.set_title(prodigy_label, fontsize=14)
    ax_prodigy.set_xlabel("Antigen", fontsize=12)
    ax_prodigy.set_ylabel("Binding Entity", fontsize=12)

    # Plot MSE heatmap
    ax_mse = fig.add_subplot(gs[1:3, 1])
    sns.heatmap(
        mse_matrix,
        cmap=mse_cmap,
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax_mse,
        linewidths=0.5
    )
    ax_mse.set_title("Pose Consistency (MSE, Å²)", fontsize=14)
    ax_mse.set_xlabel("Antigen", fontsize=12)
    ax_mse.set_ylabel("", fontsize=12)  # No y-label for this subplot

    # Add colorbars
    cbar_approx = plt.colorbar(
        plt.cm.ScalarMappable(cmap=approx_energy_cmap),
        cax=fig.add_subplot(gs[1, 2])
    )
    cbar_approx.set_label(energy_label, fontsize=10)

    cbar_prodigy = plt.colorbar(
        plt.cm.ScalarMappable(cmap=prodigy_energy_cmap),
        cax=fig.add_subplot(gs[2, 2])
    )
    cbar_prodigy.set_label(prodigy_label, fontsize=10)

    # Add explanatory text about scaling
    if scale_two_chain:
        fig.text(0.5, 0.02,
                "Note: Two-chain antibody binding energies are scaled by 0.5 for better comparison with single-chain antibodies",
                ha="center", fontsize=10, style="italic")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()