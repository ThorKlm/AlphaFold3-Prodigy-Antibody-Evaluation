"""
Matrix visualization for AlphaFold3 binding evaluation results.

This script creates a comprehensive matrix visualization showing the relationship
between binding pose consistency (MSE) and binding energy for all combinations
of antibodies and antigens.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create matrix visualization of AlphaFold3 binding evaluation results"
    )

    parser.add_argument(
        "--results", "-r", required=True,
        help="Path to combined_analysis_results.csv file"
    )

    parser.add_argument(
        "--output", "-o", default="matrix_visualization.png",
        help="Output filename for the visualization"
    )

    parser.add_argument(
        "--scale-two-chain", "-s", action="store_true",
        help="Scale binding energies of two-chain antibodies by 0.5 for normalization"
    )

    return parser.parse_args()


def detect_antibody_type(df):
    """
    Detect whether each binding entity is a single-chain or two-chain antibody.

    Args:
        df: DataFrame with binding entity information

    Returns:
        Dictionary mapping binding entities to their types (1 or 2 chains)
    """
    binding_entity_types = {}

    # Use naming convention to detect type
    # Assuming that nanobodies typically start with "nb" and antibodies with "ab"
    # You might need to adjust this logic based on your specific naming conventions
    for entity in df["Binding_Entity"].unique():
        if entity.lower().startswith("nb"):
            binding_entity_types[entity] = 1  # Single-chain
        elif entity.lower().startswith("ab"):
            binding_entity_types[entity] = 2  # Two-chain
        else:
            # For unknown naming patterns, use a heuristic based on available data
            # You might need to implement a more sophisticated detection method
            binding_entity_types[entity] = 2  # Default to two-chain

    return binding_entity_types


def normalize_binding_energies(df, binding_entity_types, scale_two_chain=True):
    """
    Normalize binding energies if needed.

    Args:
        df: DataFrame with binding energy data
        binding_entity_types: Dictionary mapping binding entities to their types
        scale_two_chain: Whether to scale binding energies of two-chain antibodies

    Returns:
        DataFrame with normalized binding energies
    """
    df_norm = df.copy()

    if scale_two_chain:
        # Scale binding energies of two-chain antibodies by 0.5
        for index, row in df_norm.iterrows():
            entity = row["Binding_Entity"]
            if binding_entity_types.get(entity, 1) == 2:
                df_norm.at[index, "Mean_Estimated_DeltaG_kcal_per_mol"] *= 0.5
                df_norm.at[index, "Is_Scaled"] = True
            else:
                df_norm.at[index, "Is_Scaled"] = False
    else:
        df_norm["Is_Scaled"] = False

    return df_norm


def create_matrix_visualization(df, output_file, scale_two_chain=True):
    """
    Create a comprehensive matrix visualization.

    Args:
        df: DataFrame with analysis results
        output_file: Output filename for the visualization
        scale_two_chain: Whether to scale binding energies of two-chain antibodies
    """
    # Detect antibody types
    binding_entity_types = detect_antibody_type(df)

    # Normalize binding energies if needed
    df_norm = normalize_binding_energies(df, binding_entity_types, scale_two_chain)

    # Get unique binding entities and antigens
    binding_entities = sorted(df_norm["Binding_Entity"].unique())
    antigens = sorted(df_norm["Antigen"].unique())

    # Create matrices for binding energy and MSE
    energy_matrix = pd.pivot_table(
        df_norm,
        values="Mean_Estimated_DeltaG_kcal_per_mol",
        index="Binding_Entity",
        columns="Antigen",
        fill_value=np.nan
    ).reindex(index=binding_entities, columns=antigens)

    mse_matrix = pd.pivot_table(
        df_norm,
        values="Center_Position_MSE",
        index="Binding_Entity",
        columns="Antigen",
        fill_value=np.nan
    ).reindex(index=binding_entities, columns=antigens)

    # Create a figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, width_ratios=[10, 10, 1], height_ratios=[1, 10])

    # Add title
    if scale_two_chain:
        plt.suptitle("AlphaFold3 Binding Evaluation Matrix\n(Two-Chain Antibody Energies Scaled by 0.5)",
                     fontsize=16, y=0.98)
    else:
        plt.suptitle("AlphaFold3 Binding Evaluation Matrix", fontsize=16, y=0.98)

    # Create custom colormaps
    energy_cmap = LinearSegmentedColormap.from_list(
        "energy_cmap", ["#ffcccc", "#ff0000", "#990000", "#550000"]
    )
    mse_cmap = LinearSegmentedColormap.from_list(
        "mse_cmap", ["#ccffcc", "#00ff00", "#009900", "#005500"]
    )

    # Plot binding energy heatmap
    ax_energy = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        energy_matrix,
        cmap=energy_cmap,
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax_energy,
        linewidths=0.5,
        center=0
    )
    ax_energy.set_title("Binding Energy (kcal/mol)", fontsize=14)
    ax_energy.set_xlabel("Antigen", fontsize=12)
    ax_energy.set_ylabel("Binding Entity", fontsize=12)

    # Add colorbar for binding energy
    ax_energy_cbar = fig.add_subplot(gs[1, 2])
    cbar_energy = plt.colorbar(
        plt.cm.ScalarMappable(cmap=energy_cmap),
        cax=ax_energy_cbar
    )
    cbar_energy.set_label("Binding Energy (kcal/mol)", fontsize=12)

    # Plot MSE heatmap
    ax_mse = fig.add_subplot(gs[1, 1])
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

    # Add colorbar for MSE
    ax_mse_cbar = fig.add_subplot(gs[0, 2])
    cbar_mse = plt.colorbar(
        plt.cm.ScalarMappable(cmap=mse_cmap),
        cax=ax_mse_cbar,
        orientation="horizontal"
    )
    cbar_mse.set_label("Pose Consistency (MSE, Å²)", fontsize=12)

    # Add binding entity type indicators
    ax_types = fig.add_subplot(gs[0, 0:2])
    ax_types.axis("off")

    # Create a bar on top showing single chain vs two chain
    type_colors = {1: "#e6f2ff", 2: "#ffe6e6"}  # Blue for single chain, Red for two chain
    for i, entity in enumerate(binding_entities):
        chain_type = binding_entity_types.get(entity, 1)
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

    # Add correlation plot
    # (This would go in the unused space if needed)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {output_file}")

    # Close figure
    plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Load results
    try:
        df = pd.read_csv(args.results)
        print(f"Loaded results from: {args.results}")
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # Check required columns
    required_columns = [
        "Binding_Entity",
        "Antigen",
        "Mean_Estimated_DeltaG_kcal_per_mol",
        "Center_Position_MSE"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in results file: {missing_columns}")
        return

    # Create visualization
    create_matrix_visualization(df, args.output, args.scale_two_chain)


if __name__ == "__main__":
    main()