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
    Create a strip plot of MSE values with error bars and individual points.

    Args:
        df: DataFrame containing MSE data
        output_path: Path to save the plot
        x_col: Column name for x-axis categories
        y_col: Column name for y-axis values
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)

    # Check if we have individual model data or summary data
    if 'Num_Models' in df.columns:
        # We have summary data with std
        std_col = y_col.replace('Mean_', 'Std_') if 'Mean_' in y_col else f"Std_{y_col}"

        if std_col in df.columns:
            # Create error bar plot
            x_positions = range(len(df))
            colors = sns.color_palette("tab10", n_colors=len(df))

            plt.errorbar(
                x_positions,
                df[y_col],
                yerr=df[std_col],
                fmt='o',
                capsize=8,
                capthick=2,
                markersize=10,
                linewidth=2,
                alpha=0.8
            )

            # Color the points
            for i, (pos, mean_val) in enumerate(zip(x_positions, df[y_col])):
                plt.scatter(pos, mean_val, color=colors[i], s=100, zorder=5)

            plt.xticks(x_positions, df[x_col], rotation=45, ha='right')
        else:
            # Fallback to simple scatter
            plt.scatter(range(len(df)), df[y_col], s=100)
            plt.xticks(range(len(df)), df[x_col], rotation=45, ha='right')
    else:
        # We have raw data points - create strip plot with summary overlay
        palette = sns.color_palette("tab10", n_colors=df[x_col].nunique())
        combo_to_color = dict(zip(df[x_col].unique(), palette))

        # Plot individual points
        sns.stripplot(data=df, x=x_col, y=y_col, jitter=True,
                      alpha=0.6, size=6, hue=x_col, palette=combo_to_color, legend=False)

        # Overlay group mean with error bars
        grouped = df.groupby(x_col)[y_col]
        means = grouped.mean()
        stds = grouped.std()
        positions = range(len(means))

        for i, combo in enumerate(means.index):
            plt.errorbar(x=i, y=means[combo], yerr=stds[combo],
                         fmt='D', color='red', capsize=8, capthick=3,
                         markersize=8, linewidth=3, zorder=10)

        plt.xticks(positions, means.index, rotation=45, ha='right')

    # Final styling
    plt.ylabel("Center-Aligned MSE (Angstrom squared)")
    plt.xlabel("Binding Entity-Antigen Combination")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_binding_energy_boxplot(df: pd.DataFrame, output_path: Union[str, Path],
                               x_col: str = "Combination",
                               y_col: str = "Estimated_DeltaG_kcal_per_mol",
                               title: str = "Estimated Binding Energy Across Seeds",
                               figsize: tuple = (10, 6)) -> None:
    """
    Create error bar plot instead of boxplot for binding energy values.

    Args:
        df: DataFrame containing binding energy data
        output_path: Path to save the plot
        x_col: Column name for x-axis categories
        y_col: Column name for y-axis values
        title: Plot title
        figsize: Figure size as (width, height) tuple
    """
    plt.figure(figsize=figsize)

    # Check if this is summary data with std columns
    std_col = y_col.replace('Mean_', 'Std_') if 'Mean_' in y_col else f"Std_{y_col}"

    if std_col in df.columns:
        # Create error bar plot with mean and std
        x_positions = range(len(df))
        colors = sns.color_palette("tab10", n_colors=len(df))

        plt.errorbar(
            x_positions,
            df[y_col],
            yerr=df[std_col],
            fmt='o',
            capsize=8,
            capthick=2,
            markersize=10,
            linewidth=2,
            alpha=0.8
        )

        # Color the points
        for i, (pos, mean_val) in enumerate(zip(x_positions, df[y_col])):
            plt.scatter(pos, mean_val, color=colors[i], s=100, zorder=5)

        plt.xticks(x_positions, df[x_col], rotation=45, ha='right')
    else:
        # If no std column, create grouped error bars
        grouped = df.groupby(x_col)[y_col]
        means = grouped.mean()
        stds = grouped.std()

        x_positions = range(len(means))
        colors = sns.color_palette("tab10", n_colors=len(means))

        plt.errorbar(
            x_positions,
            means,
            yerr=stds,
            fmt='o',
            capsize=8,
            capthick=2,
            markersize=10,
            linewidth=2
        )

        # Color the points
        for i, (pos, mean_val) in enumerate(zip(x_positions, means)):
            plt.scatter(pos, mean_val, color=colors[i], s=100, zorder=5)

        plt.xticks(x_positions, means.index, rotation=45, ha='right')

    # Final styling
    plt.ylabel("Estimated Delta G (kcal/mol)")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_energy_vs_mse(df: pd.DataFrame, output_path: Union[str, Path],
                      x_col: str = "Interface_Deviation_RMSD",
                      y_col: str = "Mean_Estimated_DeltaG_kcal_per_mol",
                      combo_col: str = "Combination",
                      title: str = "Mean Delta G vs Mean Pose Deviation per Combination",
                      figsize: tuple = (8, 6),
                      show_regression: bool = True) -> None:
    """
    Empty function - removed to simplify outputs.
    """
    print(f"Skipping energy vs MSE plot: {output_path}")
    pass


def create_confusion_matrix_heatmap(df, output_path,
                                           binding_col="Binding_Entity",
                                           antigen_col="Antigen",
                                           value_col="Mean_Estimated_DeltaG_kcal_per_mol",
                                           title="Binding Energy Matrix with Error Bars",
                                           figsize=(14, 12),
                                           cmap="YlOrRd"):
    """
    Create a heatmap with mean and standard deviation annotations.

    Args:
        df: DataFrame containing binding energy data
        output_path: Path to save the plot
        binding_col: Column name for binding entities
        antigen_col: Column name for antigens
        value_col: Column name for values (mean)
        title: Plot title
        figsize: Figure size as (width, height) tuple
        cmap: Colormap for the heatmap
    """
    # Sort the names alphabetically (case insensitive)
    binding_entities = sorted(df[binding_col].unique(), key=str.lower)
    antigens = sorted(df[antigen_col].unique(), key=str.lower)

    # Remove duplicates and pivot
    df_unique = df.drop_duplicates(subset=[binding_col, antigen_col], keep='first')

    # Create mean matrix
    matrix = df_unique.pivot(index=binding_col, columns=antigen_col, values=value_col)
    matrix = matrix.reindex(index=binding_entities, columns=antigens)

    if matrix.isna().all().all():
        print(f"Warning: All values are NaN in matrix for {value_col}")
        return

    # Make the plot
    plt.figure(figsize=figsize)

    # Determine colormap and label based on value type
    if 'MSE' in value_col or 'mse' in value_col.lower():
        cmap = 'viridis_r'
        colorbar_label = 'Center Position MSE (Angstrom squared)'
        diagonal_color = 'red'
    elif 'energy' in value_col.lower() or 'deltag' in value_col.lower():
        cmap = 'RdYlBu_r'
        colorbar_label = 'Binding Energy (kcal/mol)'
        diagonal_color = 'blue'
    elif 'score' in value_col.lower() or 'confidence' in value_col.lower():
        cmap = 'viridis'
        colorbar_label = 'Confidence Score'
        diagonal_color = 'red'
    else:
        colorbar_label = value_col.replace('_', ' ')
        diagonal_color = 'red'

    # Check for standard deviation column
    std_col = value_col.replace('Mean_', 'Std_')
    if std_col in df_unique.columns:
        # Create std matrix
        std_matrix = df_unique.pivot(index=binding_col, columns=antigen_col, values=std_col)
        std_matrix = std_matrix.reindex(index=binding_entities, columns=antigens)

        # Create combined annotations like "2.35\n±0.12"
        annot_labels = []
        for i in range(len(matrix.index)):
            row_labels = []
            for j in range(len(matrix.columns)):
                mean_val = matrix.iloc[i, j]
                std_val = std_matrix.iloc[i, j]

                if pd.isna(mean_val) or pd.isna(std_val):
                    row_labels.append("")
                else:
                    # Format with error bars
                    row_labels.append(f"{mean_val:.2f}\n±{std_val:.2f}")
            annot_labels.append(row_labels)

        # Use custom annotations
        ax = sns.heatmap(
            matrix,
            annot=annot_labels,
            fmt='',
            cmap=cmap,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': colorbar_label},
            annot_kws={'size': 14, 'ha': 'center', 'va': 'center'}
        )
    else:
        # Standard numeric annotations without error bars
        ax = sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': colorbar_label},
            annot_kws={'size': 16}
        )

    # Make colorbar label bigger
    cbar = ax.collections[0].colorbar
    cbar.set_label(colorbar_label, size=20)

    # Highlight the diagonal - these should be the real binding pairs
    for i in range(min(len(binding_entities), len(antigens))):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                  edgecolor=diagonal_color, lw=5))

    # Title and labels with exact font sizes
    plt.title(title, fontsize=22, pad=20)
    plt.xlabel('Antigens', fontsize=22)
    plt.ylabel('Binding Entities', fontsize=22)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    Empty function - removed to simplify outputs.
    """
    print(f"Skipping combined plot: {output_path}")
    pass


def create_matrix_visualization(df_energy: pd.DataFrame, df_mse: pd.DataFrame,
                                output_path: Union[str, Path],
                                scale_two_chain: bool = False,
                                title: str = "AlphaFold3 Binding Evaluation Matrix",
                                figsize: tuple = (16, 12)) -> None:
    """
    Create a comprehensive matrix visualization showing binding energy and MSE.
    FIXED to handle different MSE column names dynamically.
    """

    # FIND THE CORRECT MSE COLUMN NAME
    mse_columns = [col for col in df_mse.columns if 'MSE' in col]

    if 'Center_Position_MSE_Top' in df_mse.columns:
        mse_col = 'Center_Position_MSE_Top'
    elif 'Center_Position_MSE_Weighted' in df_mse.columns:
        mse_col = 'Center_Position_MSE_Weighted'
    elif 'Center_Position_MSE' in df_mse.columns:
        mse_col = 'Center_Position_MSE'
    elif mse_columns:
        mse_col = mse_columns[0]
    else:
        raise ValueError(f"No MSE columns found in df_mse. Available columns: {list(df_mse.columns)}")

    print(f"DEBUG: Using MSE column '{mse_col}' in matrix visualization")

    # CREATE A STANDARDIZED MSE DATAFRAME FOR MERGING
    df_mse_standard = df_mse[["Combination", "Center_Position_MSE_Top"]].rename(
        columns={"Center_Position_MSE_Top": "Center_Position_MSE"})

    # Merge dataframes using the standardized column
    df_combined = pd.merge(
        df_energy,
        df_mse_standard,  # Use standardized dataframe
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
        values="Center_Position_MSE",  # Now this will exist
        index="Binding_Entity",
        columns="Antigen",
        fill_value=np.nan
    ).reindex(index=binding_entities, columns=antigens)

    # Rest of the function remains the same...
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
    energy_label = "Approximate Delta G (kcal/mol)"
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
    prodigy_label = "PRODIGY Delta G (kcal/mol)"
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
    ax_mse.set_title(f"Pose Consistency ({mse_col}, Angstrom squared)", fontsize=14)
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