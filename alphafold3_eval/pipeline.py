"""
Pipeline for AlphaFold3 binding evaluation.
"""
"""
Pipeline for AlphaFold3 binding evaluation.
"""
import glob
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
import matplotlib.pyplot as plt
import seaborn as sns

# Core imports
from alphafold3_eval.config import Config
from alphafold3_eval.cache_manager import BindingEnergyCache

# Structure utilities
from alphafold3_eval.structure_uitls import (
    load_structure,
    extract_best_model_from_zip,
    extract_all_models_from_zip,
    parse_model_filename,
    load_alphafold2_multimer_model,
    group_af2_multimer_models,
    split_chains_by_type,
    compute_chain_center  # Make sure this matches alignment.py usage
)

# Analysis modules
from alphafold3_eval.alignment import (
    align_structure_to_reference,
    compute_antigen_aligned_mse,
    compute_weighted_mse_all_models
)

from alphafold3_eval.binding_analysis import (
    calculate_binding_energies,
    calculate_interface_rmsd,
    extract_interface_centroid,
    detect_antibody_type,
    estimate_binding_energy_simple,
    run_prodigy_binding_energy
)

# Advanced analysis (for comprehensive mode)
from alphafold3_eval.confidence_metrics import (
    extract_af3_confidence_from_cif,
    apply_confidence_calibration,
    extract_interface_plddt
)

from alphafold3_eval.interface_persistence import calculate_interface_persistence

from alphafold3_eval.ensemble_scoring import (
    calculate_ensemble_score,
    calculate_statistical_significance
)

# Visualization
from alphafold3_eval.visualization import (
    plot_mse_distributions,
    plot_binding_energy_boxplot,
    plot_energy_vs_mse,
    create_confusion_matrix_heatmap,
    create_combined_plot,
    create_matrix_visualization
)

# Create both MSE confusion matrices
def create_visualizations(df_models: pd.DataFrame, df_summary: pd.DataFrame,
                          df_mse: pd.DataFrame, config: Config, is_scaled: bool = False,
                          df_summary_top=None, df_summary_weighted=None):
    """
    Create all visualizations including 4 energy matrices.
    """
    suffix = "_scaled" if is_scaled else ""
    scale_text = " (Scaled)" if is_scaled else ""

    print(f"  Creating visualizations{scale_text}...")

    # Find the correct MSE column name
    mse_columns = [col for col in df_mse.columns if 'MSE' in col]

    if 'Center_Position_MSE_Top' in df_mse.columns:
        primary_mse_col = 'Center_Position_MSE_Top'
    elif 'Center_Position_MSE_Weighted' in df_mse.columns:
        primary_mse_col = 'Center_Position_MSE_Weighted'
    elif 'Center_Position_MSE' in df_mse.columns:
        primary_mse_col = 'Center_Position_MSE'
    elif mse_columns:
        primary_mse_col = mse_columns[0]
    else:
        print(f"Warning: No MSE columns found for visualization")
        return

    print(f"  Using MSE column '{primary_mse_col}' for visualizations")

    # 1. MSE Distribution Plot
    mse_plot_path = config.results_dir / f"mse_distribution_plot{suffix}.png"
    plot_mse_distributions(
        df_mse,
        output_path=mse_plot_path,
        y_col=primary_mse_col,
        title=f"Binding Pose Consistency Across AlphaFold3 Seeds{scale_text}"
    )

    # 2. MSE Matrix Heatmaps
    if 'Center_Position_MSE_Top' in df_mse.columns:
        create_confusion_matrix_heatmap(
            df_mse,
            output_path=config.results_dir / f"mse_matrix_top_model{suffix}.png",
            value_col="Center_Position_MSE_Top",
            title=f"Binding Pose Consistency Matrix (Top Model){scale_text}",
            cmap="viridis_r"
        )

    if 'Center_Position_MSE_Weighted' in df_mse.columns:
        create_confusion_matrix_heatmap(
            df_mse,
            output_path=config.results_dir / f"mse_matrix_weighted{suffix}.png",
            value_col="Center_Position_MSE_Weighted",
            title=f"Binding Pose Consistency Matrix (Weighted){scale_text}",
            cmap="viridis_r"
        )

    # 3. Create 4 Energy Matrix Heatmaps
    if df_summary_top is not None and df_summary_weighted is not None:

        # List of (dataframe, approx_col, prodigy_col, filename_suffix, title_suffix)
        energy_configs = [
            (df_summary_top, "Mean_Approx_DeltaG_kcal_per_mol", "Mean_Prodigy_DeltaG_kcal_per_mol", "top", "Top Model"),
            (df_summary_weighted, "Mean_Approx_DeltaG_kcal_per_mol", "Mean_Prodigy_DeltaG_kcal_per_mol", "weighted",
             "Weighted")
        ]

        for df_data, approx_col, prodigy_col, filename_suffix, title_suffix in energy_configs:

            # Approximation method matrix
            if approx_col in df_data.columns:
                matrix_path = config.results_dir / f"energy_matrix_{filename_suffix}_approx{suffix}.png"
                create_confusion_matrix_heatmap(
                    df_data,
                    output_path=matrix_path,
                    value_col=approx_col,
                    title=f"Binding Energy Matrix - {title_suffix} Approximation{scale_text}",
                    cmap="YlOrRd"
                )

            # PRODIGY method matrix
            if prodigy_col in df_data.columns and not df_data[prodigy_col].isna().all():
                matrix_path = config.results_dir / f"energy_matrix_{filename_suffix}_prodigy{suffix}.png"
                create_confusion_matrix_heatmap(
                    df_data,
                    output_path=matrix_path,
                    value_col=prodigy_col,
                    title=f"Binding Energy Matrix - {title_suffix} PRODIGY{scale_text}",
                    cmap="YlOrRd"
                )

    # 4. Original energy plot (keep for backwards compatibility)
    if 'Mean_Approx_DeltaG_kcal_per_mol' in df_summary.columns:
        energy_plot_path = config.results_dir / f"binding_energy_plot{suffix}.png"

        plt.figure(figsize=(12, 6))

        if 'Std_Approx_DeltaG_kcal_per_mol' in df_summary.columns:
            x_positions = range(len(df_summary))
            plt.errorbar(
                x_positions,
                df_summary['Mean_Approx_DeltaG_kcal_per_mol'],
                yerr=df_summary['Std_Approx_DeltaG_kcal_per_mol'],
                fmt='o',
                capsize=5,
                capthick=2,
                markersize=8
            )
            plt.xticks(x_positions, df_summary['Combination'], rotation=45, ha='right')
        else:
            plt.scatter(range(len(df_summary)), df_summary['Mean_Approx_DeltaG_kcal_per_mol'])
            plt.xticks(range(len(df_summary)), df_summary['Combination'], rotation=45, ha='right')

        plt.ylabel("Estimated ΔG (kcal/mol)")
        plt.title(f"Estimated Binding Energy Across Seeds{scale_text}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(energy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Energy vs MSE scatter plot
        df_mse_for_merge = df_mse[["Combination", primary_mse_col]].copy()
        df_mse_for_merge = df_mse_for_merge.rename(columns={primary_mse_col: "Center_Position_MSE"})

        df_combined = pd.merge(
            df_summary,
            df_mse_for_merge,
            on="Combination",
            how="left"
        )

        combined_plot_path = config.results_dir / f"energy_vs_mse_plot{suffix}.png"
        plot_energy_vs_mse(
            df_combined,
            output_path=combined_plot_path,
            x_col="Center_Position_MSE",
            y_col="Mean_Approx_DeltaG_kcal_per_mol",
            title=f"Mean ΔG vs Pose Consistency{scale_text}"
        )

    print(f"  Visualizations{scale_text} completed")


# Fix for pipeline.py run_pipeline function
# The issue is in the merge operation where we're looking for 'Center_Position_MSE'
# but the analyze_mse_with_cache function returns different column names

def run_pipeline(input_dir, config, use_prodigy=True,
                 clean_intermediate=False, input_format="alphafold3"):
    """
    Run the complete analysis pipeline.
    """
    print("Starting AlphaFold binding evaluation pipeline...")
    print("  Input directory: " + str(input_dir))
    print("  Output directory: " + str(config.results_dir))
    print("  Using PRODIGY: " + str(use_prodigy))
    print("  Input format: " + input_format)
    config.analysis_config["min_seeds"] = 2

    # Step 1: Extract models
    combination_to_models = extract_models(input_dir, config, input_format)
    if not combination_to_models:
        raise ValueError("No valid models extracted from " + str(input_dir) + " using format " + input_format)

    # Step 2: Analyze MSE
    df_mse = analyze_mse_with_cache(combination_to_models, config)

    # Step 3: Analyze binding energy (now returns 6 dataframes)
    df_models, df_models_scaled, df_summary_top, df_summary_top_scaled, df_summary_weighted, df_summary_weighted_scaled = analyze_binding_energy_with_cache(
        combination_to_models, config, use_prodigy)

    # Step 4: Find correct MSE column for merging
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
        raise ValueError(f"No MSE columns found in dataframe")

    # Create standardized MSE column for merging
    df_mse_for_merge = df_mse[["Combination", mse_col]].copy()
    df_mse_for_merge = df_mse_for_merge.rename(columns={mse_col: "Center_Position_MSE"})

    # Step 5: Save combined results (using weighted summary for backwards compatibility)
    df_combined = pd.merge(df_summary_weighted, df_mse_for_merge, on="Combination", how="outer")
    df_combined.to_csv(config.output_files["combined_results"], index=False)
    print("  Combined results saved to: " + str(config.output_files["combined_results"]))

    df_combined_scaled = pd.merge(df_summary_weighted_scaled, df_mse_for_merge, on="Combination", how="outer")
    scaled_path = str(config.output_files["combined_results"]).replace(".csv", "_scaled.csv")
    df_combined_scaled.to_csv(scaled_path, index=False)
    print("  Scaled combined results saved to: " + scaled_path)

    # Step 6: Create visualizations with all 4 energy matrices
    create_visualizations(
        df_models,
        df_summary_weighted,  # Use weighted for backward compatibility
        df_mse,
        config,
        is_scaled=False,
        df_summary_top=df_summary_top,
        df_summary_weighted=df_summary_weighted
    )

    # Step 7: Clean up if requested
    if clean_intermediate:
        cleanup_intermediate_files(config)

    print("Pipeline completed successfully!")


def run_mse_only_pipeline(input_dir, config, input_format="alphafold3"):
    """
    Run MSE-only analysis with consistent file naming.
    """
    print("Starting MSE-only analysis pipeline...")
    print("  Input directory: " + str(input_dir))
    print("  Output directory: " + str(config.results_dir))
    print("  Input format: " + input_format)

    # Step 1: Extract models
    combination_to_models = extract_models(input_dir, config, input_format)

    if not combination_to_models:
        raise ValueError("No valid models extracted from " + str(input_dir))

    # Step 2: Analyze MSE only
    df_mse = analyze_mse_with_cache(combination_to_models, config)

    # Step 3: Create MSE visualizations with consistent naming

    # Find the correct MSE column
    mse_columns = [col for col in df_mse.columns if 'MSE' in col]
    if 'Center_Position_MSE_Top' in df_mse.columns:
        primary_mse_col = 'Center_Position_MSE_Top'
    elif 'Center_Position_MSE_Weighted' in df_mse.columns:
        primary_mse_col = 'Center_Position_MSE_Weighted'
    elif 'Center_Position_MSE' in df_mse.columns:
        primary_mse_col = 'Center_Position_MSE'
    elif mse_columns:
        primary_mse_col = mse_columns[0]
    else:
        raise ValueError("No MSE columns found!")

    # MSE distribution plot
    plot_mse_distributions(
        df_mse,
        output_path=config.results_dir / "mse_distribution_plot.png",
        y_col=primary_mse_col,
        title="Binding Pose Consistency Across AlphaFold3 Seeds"
    )

    # MSE matrix heatmaps
    if 'Center_Position_MSE_Top' in df_mse.columns:
        create_confusion_matrix_heatmap(
            df_mse,
            output_path=config.results_dir / "mse_matrix_top_model.png",
            value_col="Center_Position_MSE_Top",
            title="Binding Pose Consistency Matrix (Top Model)",
            cmap="viridis_r"
        )

    if 'Center_Position_MSE_Weighted' in df_mse.columns:
        create_confusion_matrix_heatmap(
            df_mse,
            output_path=config.results_dir / "mse_matrix_weighted.png",
            value_col="Center_Position_MSE_Weighted",
            title="Binding Pose Consistency Matrix (Weighted)",
            cmap="viridis_r"
        )

    print("\nMSE-only analysis completed successfully!")
    print("  MSE distribution plot saved to: " + str(config.results_dir / "mse_distribution_plot.png"))
    print("  MSE matrix plots saved to results directory")

def analyze_binding_energy(combination_to_models, config, use_prodigy=True):
    """
    Analyze binding energy for each model using both methods.

    Args:
        combination_to_models: Dictionary mapping combinations to lists of model paths
        config: Configuration object
        use_prodigy: Whether to use PRODIGY for binding energy calculation

    Returns:
        Tuple of (unscaled_df, scaled_df) with binding energy results
    """
    print("Analyzing binding energy...")

    # Create temp directory for PRODIGY PDB files
    # prodigy_temp_dir = os.path.join(config.temp_dir, "prodigy_pdbs")
    prodigy_temp_dir = os.path.normpath(os.path.join(str(config.temp_dir), "prodigy_pdbs"))
    os.makedirs(prodigy_temp_dir, exist_ok=True)

    records = []

    for combination, model_paths in combination_to_models.items():
        print("  Computing binding energy for " + combination)

        # Parse combination
        parts = combination.split('_')
        if len(parts) >= 3:
            binding_entity = parts[0] + '_' + parts[1]
            antigen = parts[2]
        else:
            binding_entity = parts[0]
            antigen = parts[1]

        # Create reference for alignment
        reference_structure = None

        # Process each model
        for model_path in model_paths:
            try:
                # Parse metadata
                meta = parse_model_filename(model_path)
                if not meta:
                    continue

                # Load structure based on file extension
                structure = load_structure(model_path)
                if not structure:
                    continue

                # Store binding entity name in structure for antibody type detection
                structure.binding_entity = binding_entity

                # Align to reference if available
                if reference_structure is None:
                    reference_structure = structure
                else:
                    structure = align_structure_to_reference(structure, reference_structure)

                # Compute interface centroid
                centroid = extract_interface_centroid(structure, cutoff=config.analysis_config["contact_cutoff"])

                # Configuration for binding energy calculation
                binding_config = {
                    'temperature': config.prodigy_config["temperature"],
                    'contact_cutoff': config.analysis_config["contact_cutoff"],
                    'temp_dir': prodigy_temp_dir
                }

                # Compute binding energies using both methods
                energy_results = calculate_binding_energies(
                    structure,
                    binding_config,
                    use_prodigy=use_prodigy
                )

                # Detect antibody type
                antibody_type = detect_antibody_type(binding_entity)

                # Create record
                record = {
                    "Combination": combination,
                    "Binding_Entity": binding_entity,
                    "Antigen": antigen,
                    "Seed": meta["seed"],
                    "Approx_DeltaG_kcal_per_mol": energy_results["approx_energy"],
                    "Prodigy_DeltaG_kcal_per_mol": energy_results["prodigy_energy"],
                    "Contacts": energy_results["contacts"],
                    "Interface_Centroid": centroid,
                    "Model_Path": model_path,
                    "Antibody_Type": antibody_type
                }

                # Add record
                records.append(record)

            except Exception as e:
                print("  Error processing " + model_path + ": " + str(e))

    # Create DataFrame
    df_models = pd.DataFrame(records)

    if df_models.empty:
        raise ValueError("No valid models processed - check file integrity or parsing.")

    # Create normalized versions with scaling for two-chain antibodies
    df_models_scaled = df_models.copy()
    for index, row in df_models_scaled.iterrows():
        if row["Antibody_Type"] == 2:
            if pd.notna(row["Approx_DeltaG_kcal_per_mol"]):
                df_models_scaled.at[index, "Approx_DeltaG_kcal_per_mol"] *= 0.5
            if pd.notna(row["Prodigy_DeltaG_kcal_per_mol"]):
                df_models_scaled.at[index, "Prodigy_DeltaG_kcal_per_mol"] *= 0.5

    df_models_scaled["Is_Scaled"] = df_models_scaled["Antibody_Type"] == 2
    df_models["Is_Scaled"] = False

    # Save results (both unscaled and scaled)
    # Remove centroid column before saving (not serializable)
    df_save = df_models.copy()
    if "Interface_Centroid" in df_save.columns:
        df_save = df_save.drop(columns=["Interface_Centroid"])

    df_save_scaled = df_models_scaled.copy()
    if "Interface_Centroid" in df_save_scaled.columns:
        df_save_scaled = df_save_scaled.drop(columns=["Interface_Centroid"])

    # Save unscaled results
    df_save.to_csv(config.output_files["binding_energy"], index=False)
    print("  Binding energy results saved to: " + str(config.output_files["binding_energy"]))

    # Save scaled results
    scaled_path = str(config.output_files["binding_energy"]).replace(".csv", "_scaled.csv")
    df_save_scaled.to_csv(scaled_path, index=False)
    print("  Scaled binding energy results saved to: " + scaled_path)

    # Clean up temporary PDB files
    shutil.rmtree(prodigy_temp_dir, ignore_errors=True)

    return df_models, df_models_scaled


# Update the analyze_binding_energy function to use caching
def analyze_binding_energy_with_cache(combination_to_models: Dict[str, List[str]],
                                      config: Config,
                                      use_prodigy: bool = True) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze binding energy with caching support.
    Returns both individual model data and summary dataframes for top model and weighted approaches.
    """
    print("Analyzing binding energy with caching...")

    # Initialize cache
    cache = BindingEnergyCache(config.temp_dir / "binding_energy_cache")
    cache_stats = cache.get_cache_stats()
    print(f"  Cache statistics: {cache_stats}")

    # Create temp directory for PRODIGY PDB files
    prodigy_temp_dir = os.path.join(config.temp_dir, "prodigy_pdbs")
    os.makedirs(prodigy_temp_dir, exist_ok=True)

    records = []
    cached_count = 0
    computed_count = 0

    for combination, model_paths in combination_to_models.items():
        print(f"  Computing binding energy for {combination}")

        # Parse combination
        parts = combination.split('_')
        if len(parts) >= 3:
            binding_entity = parts[0] + '_' + parts[1]
            antigen = parts[2]
        else:
            binding_entity = parts[0]
            antigen = parts[1]

        for model_path in model_paths:
            try:
                # Parse metadata to get model rank
                meta = parse_model_filename(model_path)
                if not meta:
                    continue

                # Extract model rank from filename (e.g., "_model_0.cif" -> rank 0)
                model_rank = 0
                if "_model_" in model_path:
                    try:
                        model_rank = int(model_path.split("_model_")[1].split(".")[0])
                    except:
                        model_rank = 0

                # Check cache for simple binding energy
                cached_simple = cache.get_cached_energy(
                    model_path,
                    method='simple',
                    temperature=config.prodigy_config["temperature"],
                    cutoff=config.analysis_config["contact_cutoff"]
                )

                # Check cache for PRODIGY if needed
                cached_prodigy = None
                if use_prodigy:
                    cached_prodigy = cache.get_cached_energy(
                        model_path,
                        method='prodigy',
                        temperature=config.prodigy_config["temperature"],
                        cutoff=config.analysis_config["contact_cutoff"]
                    )

                # Use cached values if available
                if cached_simple and (not use_prodigy or cached_prodigy):
                    # All required values are cached
                    approx_energy = cached_simple.get('energy', 0.0)
                    contacts = cached_simple.get('contacts', 0)
                    prodigy_energy = cached_prodigy.get('energy', None) if cached_prodigy else None
                    cached_count += 1

                elif cached_simple and not use_prodigy:
                    # Simple energy cached and PRODIGY not needed
                    approx_energy = cached_simple.get('energy', 0.0)
                    contacts = cached_simple.get('contacts', 0)
                    prodigy_energy = None
                    cached_count += 1

                else:
                    # Need to compute some or all values
                    # Load structure only if needed
                    structure = load_structure(model_path)
                    if not structure:
                        continue

                    structure.binding_entity = binding_entity

                    # Compute simple binding energy if not cached
                    if not cached_simple:
                        approx_energy, contacts = estimate_binding_energy_simple(
                            structure,
                            cutoff=config.analysis_config["contact_cutoff"]
                        )
                        # Cache the result
                        cache.cache_energy(
                            model_path,
                            {'energy': approx_energy, 'contacts': contacts},
                            method='simple',
                            temperature=config.prodigy_config["temperature"],
                            cutoff=config.analysis_config["contact_cutoff"]
                        )
                    else:
                        approx_energy = cached_simple.get('energy', 0.0)
                        contacts = cached_simple.get('contacts', 0)

                    # Compute PRODIGY if needed and not cached
                    prodigy_energy = None
                    if use_prodigy and not cached_prodigy:
                        prodigy_energy = run_prodigy_binding_energy(
                            structure,
                            temp=config.prodigy_config["temperature"],
                            temp_dir=prodigy_temp_dir
                        )
                        if prodigy_energy is not None:
                            # Cache the result
                            cache.cache_energy(
                                model_path,
                                {'energy': prodigy_energy},
                                method='prodigy',
                                temperature=config.prodigy_config["temperature"],
                                cutoff=config.analysis_config["contact_cutoff"]
                            )
                    elif cached_prodigy:
                        prodigy_energy = cached_prodigy.get('energy', None)

                    computed_count += 1

                # Detect antibody type
                antibody_type = detect_antibody_type(binding_entity)

                # Create record with model rank info
                record = {
                    "Combination": combination,
                    "Binding_Entity": binding_entity,
                    "Antigen": antigen,
                    "Seed": meta["seed"],
                    "Model_Rank": model_rank,
                    "Approx_DeltaG_kcal_per_mol": approx_energy,
                    "Prodigy_DeltaG_kcal_per_mol": prodigy_energy,
                    "Contacts": contacts,
                    "Model_Path": model_path,
                    "Antibody_Type": antibody_type
                }

                records.append(record)

            except Exception as e:
                print(f"  Error processing {model_path}: {e}")

    print(f"  Used {cached_count} cached results, computed {computed_count} new results")

    # Create main dataframes
    df_models = pd.DataFrame(records)
    if df_models.empty:
        raise ValueError("No valid models processed")

    # Create scaled versions
    df_models_scaled = df_models.copy()
    for index, row in df_models_scaled.iterrows():
        if row["Antibody_Type"] == 2:
            if pd.notna(row["Approx_DeltaG_kcal_per_mol"]):
                df_models_scaled.at[index, "Approx_DeltaG_kcal_per_mol"] *= 0.5
            if pd.notna(row["Prodigy_DeltaG_kcal_per_mol"]):
                df_models_scaled.at[index, "Prodigy_DeltaG_kcal_per_mol"] *= 0.5

    df_models_scaled["Is_Scaled"] = df_models_scaled["Antibody_Type"] == 2
    df_models["Is_Scaled"] = False

    # Create summary dataframes for different approaches
    # Top model only (rank 0)
    df_top_models = df_models[df_models['Model_Rank'] == 0]
    df_top_models_scaled = df_models_scaled[df_models_scaled['Model_Rank'] == 0]

    df_summary_top = create_summary_dataframe(df_top_models)
    df_summary_top_scaled = create_summary_dataframe(df_top_models_scaled)

    # Weighted (all models) - same as original approach
    df_summary_weighted = create_summary_dataframe(df_models)
    df_summary_weighted_scaled = create_summary_dataframe(df_models_scaled)

    # Save results
    df_models.to_csv(config.output_files["binding_energy"], index=False)
    scaled_path = str(config.output_files["binding_energy"]).replace(".csv", "_scaled.csv")
    df_models_scaled.to_csv(scaled_path, index=False)

    # Clean up
    shutil.rmtree(prodigy_temp_dir, ignore_errors=True)

    print("analyze_binding_energy_with_cache", "returning: df_models, df_models_scaled, df_summary_top, df_summary_top_scaled, df_summary_weighted, df_summary_weighted_scaled")
    return df_models, df_models_scaled, df_summary_top, df_summary_top_scaled, df_summary_weighted, df_summary_weighted_scaled

# Continue in alphafold3_eval/pipeline.py

def run_comprehensive_analysis(
        input_dir: str,
        config: Config,
        methods: List[str] = None,
        input_format: str = "alphafold3"
) -> None:
    """
    Run comprehensive analysis with multiple evaluation methods.

    Args:
        input_dir: Directory containing input files
        config: Configuration object
        methods: List of methods to use (default: all)
        input_format: Format of input files
    """
    if methods is None:
        methods = ['mse', 'simple_energy', 'prodigy_energy', 'persistence', 'ensemble']

    print(f"Starting comprehensive analysis with methods: {methods}")

    # Step 1: Extract models
    combination_to_models = extract_models(input_dir, config, input_format)

    if not combination_to_models:
        raise ValueError(f"No valid models extracted from {input_dir}")

    # Initialize results storage
    all_results = {}

    # Step 2: Load structures once for each combination
    combination_to_structures = {}
    for combination, model_paths in combination_to_models.items():
        structures = []
        for model_path in model_paths:
            structure = load_structure(model_path)
            if structure:
                structures.append((model_path, structure))
        combination_to_structures[combination] = structures

    # Step 3: Run MSE analysis if requested
    if 'mse' in methods:
        print("\nAnalyzing binding pose consistency (MSE)...")
        mse_results = compute_weighted_mse_all_models({combination: model_paths})
        all_results['mse'] = mse_results

    # Step 4: Run binding energy analyses
    if 'simple_energy' in methods or 'prodigy_energy' in methods:
        use_prodigy = 'prodigy_energy' in methods
        print(f"\nAnalyzing binding energy (PRODIGY={use_prodigy})...")
        df_energy, df_energy_scaled = analyze_binding_energy_with_cache(
            combination_to_models, config, use_prodigy
        )
        all_results['binding_energy'] = df_energy
        all_results['binding_energy_scaled'] = df_energy_scaled

    # Step 5: Run interface persistence analysis if requested
    if 'persistence' in methods:
        print("\nAnalyzing interface persistence...")
        persistence_results = analyze_interface_persistence(
            combination_to_structures, config
        )
        all_results['persistence'] = persistence_results

    # Step 6: Extract confidence metrics
    if 'ensemble' in methods or 'confidence' in methods:
        print("\nExtracting confidence metrics...")
        confidence_results = analyze_confidence_metrics(
            combination_to_models, config
        )
        all_results['confidence'] = confidence_results

    # Step 7: Calculate ensemble scores if requested
    if 'ensemble' in methods:
        print("\nCalculating ensemble scores...")
        ensemble_results = calculate_ensemble_scores_for_all(
            all_results, combination_to_models, config
        )
        all_results['ensemble'] = ensemble_results

    # Step 8: Statistical analysis
    print("\nPerforming statistical analysis...")
    stat_results = perform_statistical_analysis(all_results, config)

    # Step 9: Create comprehensive visualizations
    create_comprehensive_visualizations(all_results, stat_results, config)

    # Step 10: Save comprehensive report
    save_comprehensive_report(all_results, stat_results, config)

    print("\nComprehensive analysis completed!")


def analyze_interface_persistence(
        combination_to_structures: Dict[str, List[Tuple[str, Structure]]],
        config: Config
) -> pd.DataFrame:
    """
    Analyze interface persistence for all combinations.
    """
    results = []

    for combination, structure_list in combination_to_structures.items():
        if len(structure_list) < 2:
            continue

        print(f"  Analyzing persistence for {combination}")

        # Extract structures only
        structures = [s[1] for s in structure_list]

        # Align all structures to first one
        aligned_structures = [structures[0]]
        for i in range(1, len(structures)):
            aligned = align_structure_to_reference(structures[i], structures[0])
            if aligned:
                aligned_structures.append(aligned)

        # Calculate persistence metrics
        persistence_metrics = calculate_interface_persistence(
            aligned_structures,
            cutoff=config.analysis_config["contact_cutoff"]
        )

        # Parse combination
        parts = combination.split('_')
        if len(parts) >= 3:
            binding_entity = f"{parts[0]}_{parts[1]}"
            antigen = parts[2]
        else:
            binding_entity = parts[0]
            antigen = parts[1]

        # Create result record
        result = {
            'Combination': combination,
            'Binding_Entity': binding_entity,
            'Antigen': antigen,
            'Mean_Persistence': persistence_metrics['mean_persistence'],
            'Contact_Conservation': persistence_metrics['contact_conservation'],
            'Persistent_Contacts': persistence_metrics['persistent_contacts'],
            'Total_Contacts': persistence_metrics['total_unique_contacts'],
            'Num_Seeds': len(aligned_structures)
        }

        results.append(result)

    df = pd.DataFrame(results)

    # Save results
    output_path = config.results_dir / "interface_persistence_results.csv"
    df.to_csv(output_path, index=False)
    print(f"  Persistence results saved to: {output_path}")

    return df

def extract_models(input_dir, config, input_format="alphafold3"):
    """
    Extract best models from input files based on format.

    Args:
        input_dir: Directory containing input files
        config: Configuration object
        input_format: Format of input files ("alphafold3" or "alphafold2_multimer")

    Returns:
        Dictionary mapping combinations to lists of model paths
    """
    if input_format.lower() == "alphafold2_multimer":
        return extract_models_af2_multimer(input_dir, config)
    else:
        return extract_models_af3_zip(input_dir, config)


def extract_models_af3_zip(input_dir, config):
    """
    Extract best models from AlphaFold3 ZIP files.

    Args:
        input_dir: Directory containing ZIP files
        config: Configuration object

    Returns:
        Dictionary mapping combinations to lists of model paths
    """
    print("Extracting best models from AlphaFold3 ZIP files...")

    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    if not zip_files:
        raise ValueError("No ZIP files found in " + str(input_dir))

    # Handle special case filenames
    renamed_files = {}
    for f in zip_files:
        if "seed_4B" in f:
            new_f = f.replace("seed_4B", "seed_4b")
            os.rename(f, new_f)
            renamed_files[new_f] = f

    # Reload zip files after renaming
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))

    # Group ZIP files by combination
    combination_to_zips = {}
    for zip_file in zip_files:
        basename = os.path.basename(zip_file)
        parts = basename.split('_')

        # Extract combination info
        if len(parts) >= 4:
            binding_entity = parts[1] + '_' + parts[2]
            antigen = parts[3]
            combination = binding_entity + "_" + antigen

            if combination not in combination_to_zips:
                combination_to_zips[combination] = []

            combination_to_zips[combination].append(zip_file)

    # Extract best models for each combination
    combination_to_models = {}

    for combination, zip_list in combination_to_zips.items():
        print("  Processing " + combination + " (" + str(len(zip_list)) + " seeds)")

        if len(zip_list) < config.analysis_config["min_seeds"]:
            print("  Warning: " + combination + " has only " + str(len(zip_list)) +
                  " seeds, minimum required is " + str(config.analysis_config["min_seeds"]))
            continue

        model_paths = []
        for zip_path in zip_list:
            # Extract model
            output_filename = Path(zip_path).stem + "_model_0.cif"
            '''model_path = extract_best_model_from_zip(
                zip_path,
                config.model_dir,
                output_filename
            )'''
            extracted_models = extract_all_models_from_zip(zip_path, config.model_dir)
            model_paths.extend(extracted_models)

            if model_path:
                model_paths.append(model_path)

        if model_paths:
            combination_to_models[combination] = model_paths

    # Restore original filenames
    for new_f, orig_f in renamed_files.items():
        os.rename(new_f, orig_f)

    return combination_to_models


def extract_models_af2_multimer(input_dir, config):
    """
    Extract models from AlphaFold2-Multimer directory structure.

    Args:
        input_dir: Directory containing AlphaFold2-Multimer predictions
        config: Configuration object

    Returns:
        Dictionary mapping combinations to lists of model paths
    """
    print("Finding AlphaFold2-Multimer models...")

    # Find all model files
    model_paths = load_alphafold2_multimer_model(
        input_dir,
        subfolder_pattern="*_seed*",
        model_file="top_model.pdb"
    )

    if not model_paths:
        # Try alternate model naming
        model_paths = load_alphafold2_multimer_model(
            input_dir,
            subfolder_pattern="*_seed*",
            model_file="ranked_0.pdb"
        )

    if not model_paths:
        raise ValueError("No AlphaFold2-Multimer models found in " + str(input_dir))

    # Group models by combination
    combination_to_models = group_af2_multimer_models(model_paths)

    # Filter combinations with too few seeds
    filtered_combinations = {}
    for combination, model_list in combination_to_models.items():
        if len(model_list) < config.analysis_config["min_seeds"]:
            print("  Warning: " + combination + " has only " + str(len(model_list)) +
                  " seeds, minimum required is " + str(config.analysis_config["min_seeds"]))
            continue

        print("  Processing " + combination + " (" + str(len(model_list)) + " seeds)")
        filtered_combinations[combination] = model_list

    return filtered_combinations


def analyze_mse_with_cache(combination_to_models: Dict[str, List[str]], config: Config) -> pd.DataFrame:
    """Analyze MSE with both top-model and weighted approaches."""

    results = []

    for combination, model_paths in combination_to_models.items():
        # Parse combination
        parts = combination.split('_')
        if len(parts) >= 3:
            binding_entity = parts[0] + "_" + parts[1]
            antigen = parts[2]
        else:
            binding_entity = parts[0]
            antigen = parts[1]

        # Calculate both MSE types
        mse_results = mse_results = compute_weighted_mse_all_models(model_paths)

        antibody_type = detect_antibody_type(binding_entity)

        results.append({
            'Binding_Entity': binding_entity,
            'Antigen': antigen,
            'Combination': combination,
            'Center_Position_MSE_Top': mse_results['top_model_mse'],
            'Center_Position_MSE_Weighted': mse_results['weighted_mse'],
            'Num_Models': len(model_paths),
            'Antibody_Type': antibody_type
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(config.output_files["mse_results"], index=False)

    return results_df


# Fix the analyze_binding_energy_with_cache function in pipeline.py

def analyze_confidence_metrics(
        combination_to_models: Dict[str, List[str]],
        config: Config
) -> pd.DataFrame:
    """
    Extract and analyze confidence metrics for all models.
    """
    results = []

    for combination, model_paths in combination_to_models.items():
        print(f"  Extracting confidence metrics for {combination}")

        # Parse combination
        parts = combination.split('_')
        if len(parts) >= 3:
            binding_entity = f"{parts[0]}_{parts[1]}"
            antigen = parts[2]
        else:
            binding_entity = parts[0]
            antigen = parts[1]

        for model_path in model_paths:
            # Extract metadata
            meta = parse_model_filename(model_path)
            if not meta:
                continue

            # Extract confidence metrics
            if model_path.endswith('.cif'):
                raw_metrics = extract_af3_confidence_from_cif(model_path)
                calibrated_metrics = apply_confidence_calibration(raw_metrics)

                # Also get interface pLDDT
                structure = load_structure(model_path)
                if structure:
                    interface_plddt = extract_interface_plddt(
                        structure,
                        config.analysis_config["contact_cutoff"]
                    )
                    calibrated_metrics['interface_pLDDT'] = interface_plddt

                result = {
                    'Combination': combination,
                    'Binding_Entity': binding_entity,
                    'Antigen': antigen,
                    'Seed': meta["seed"],
                    'Model_Path': model_path,
                    **raw_metrics,
                    **calibrated_metrics
                }

                results.append(result)

    df = pd.DataFrame(results)

    # Save results
    output_path = config.results_dir / "confidence_metrics_results.csv"
    df.to_csv(output_path, index=False)
    print(f"  Confidence metrics saved to: {output_path}")

    return df


def calculate_ensemble_scores_for_all(
        all_results: Dict,
        combination_to_models: Dict[str, List[str]],
        config: Config
) -> pd.DataFrame:
    """
    Calculate ensemble scores for all combinations.
    """
    ensemble_results = []

    # Get all combinations
    combinations = list(combination_to_models.keys())

    for combination in combinations:
        # Parse combination
        parts = combination.split('_')
        if len(parts) >= 3:
            binding_entity = f"{parts[0]}_{parts[1]}"
            antigen = parts[2]
        else:
            binding_entity = parts[0]
            antigen = parts[1]

        # Collect metrics for this combination
        metrics = {}

        # MSE (structural consistency)
        if 'mse' in all_results:
            mse_df = all_results['mse']
            mse_row = mse_df[mse_df['Combination'] == combination]
            if not mse_row.empty:
                metrics['structural_consistency'] = mse_row['Center_Position_MSE'].iloc[0]

        # Interface persistence
        if 'persistence' in all_results:
            persist_df = all_results['persistence']
            persist_row = persist_df[persist_df['Combination'] == combination]
            if not persist_row.empty:
                metrics['interface_persistence'] = persist_row['Mean_Persistence'].iloc[0]

        # Confidence (mean of calibrated ipTM)
        if 'confidence' in all_results:
            conf_df = all_results['confidence']
            conf_rows = conf_df[conf_df['Combination'] == combination]
            if not conf_rows.empty:
                mean_conf = conf_rows['calibrated_ipTM'].mean()
                metrics['confidence'] = mean_conf

        # Binding energy (mean)
        if 'binding_energy' in all_results:
            energy_df = all_results['binding_energy']
            energy_rows = energy_df[energy_df['Combination'] == combination]
            if not energy_rows.empty:
                mean_energy = energy_rows['Approx_DeltaG_kcal_per_mol'].mean()
                metrics['binding_energy'] = mean_energy

        # Calculate ensemble score
        if metrics:
            ensemble_score, normalized_scores = calculate_ensemble_score(metrics)

            result = {
                'Combination': combination,
                'Binding_Entity': binding_entity,
                'Antigen': antigen,
                'Ensemble_Score': ensemble_score,
                **{f'Score_{k}': v for k, v in normalized_scores.items()},
                **{f'Raw_{k}': v for k, v in metrics.items()}
            }

            ensemble_results.append(result)

    df = pd.DataFrame(ensemble_results)

    # Save results
    output_path = config.results_dir / "ensemble_scores.csv"
    df.to_csv(output_path, index=False)
    print(f"  Ensemble scores saved to: {output_path}")

    return df


def perform_statistical_analysis(all_results: Dict, config: Config) -> Dict:
    """
    Perform statistical analysis on all results.
    """
    stat_results = {}

    # For each scoring method, create matrix and test significance
    scoring_methods = {
        'mse': ('Center_Position_MSE', True),  # inverse=True because lower is better
        'simple_energy': ('Mean_Approx_DeltaG_kcal_per_mol', False),
        'persistence': ('Mean_Persistence', False),
        'ensemble': ('Ensemble_Score', False)
    }

    for method, (value_col, inverse) in scoring_methods.items():
        if method == 'simple_energy' and 'binding_energy' in all_results:
            df = create_summary_dataframe(all_results['binding_energy'])
        elif method in all_results:
            df = all_results[method]
        else:
            continue

        # Create matrix
        matrix = df.pivot_table(
            index='Binding_Entity',
            columns='Antigen',
            values=value_col
        ).values

        # Normalize if needed for statistical test
        if inverse:
            matrix = -matrix  # Flip sign so higher is better

        # Perform statistical test
        stats = calculate_statistical_significance(matrix)
        stat_results[method] = stats

        print(f"\nStatistical analysis for {method}:")
        print(f"  Diagonal mean: {stats['diagonal_mean']:.3f} ± {stats['diagonal_std']:.3f}")
        print(f"  Off-diagonal mean: {stats['off_diagonal_mean']:.3f} ± {stats['off_diagonal_std']:.3f}")
        print(f"  Difference: {stats['observed_difference']:.3f} (p={stats['p_value']:.4f})")
        print(f"  Effect size: {stats['effect_size']:.3f}")

    # Save statistical results
    stat_df = pd.DataFrame(stat_results).T
    stat_df.to_csv(config.results_dir / "statistical_analysis.csv")

    return stat_results


def create_summary_dataframe(df_models: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for binding energy results.

    Args:
        df_models: DataFrame with individual model results

    Returns:
        DataFrame with summary statistics per combination
    """
    if df_models.empty:
        return pd.DataFrame()

    # Group by combination and calculate statistics
    summary_stats = []

    for combination in df_models['Combination'].unique():
        combo_data = df_models[df_models['Combination'] == combination]

        # Basic info
        binding_entity = combo_data['Binding_Entity'].iloc[0]
        antigen = combo_data['Antigen'].iloc[0]
        antibody_type = combo_data['Antibody_Type'].iloc[0]
        num_models = len(combo_data)

        # Approximate binding energy stats
        approx_energies = combo_data['Approx_DeltaG_kcal_per_mol'].dropna()
        if len(approx_energies) > 0:
            mean_approx = approx_energies.mean()
            std_approx = approx_energies.std()
        else:
            mean_approx = np.nan
            std_approx = np.nan

        # PRODIGY binding energy stats
        prodigy_energies = combo_data['Prodigy_DeltaG_kcal_per_mol'].dropna()
        if len(prodigy_energies) > 0:
            mean_prodigy = prodigy_energies.mean()
            std_prodigy = prodigy_energies.std()
        else:
            mean_prodigy = np.nan
            std_prodigy = np.nan

        # Contact stats
        contacts = combo_data['Contacts'].dropna()
        if len(contacts) > 0:
            mean_contacts = contacts.mean()
            std_contacts = contacts.std()
        else:
            mean_contacts = np.nan
            std_contacts = np.nan

        summary_stats.append({
            'Combination': combination,
            'Binding_Entity': binding_entity,
            'Antigen': antigen,
            'Antibody_Type': antibody_type,
            'Num_Models': num_models,
            'Mean_Approx_DeltaG_kcal_per_mol': mean_approx,
            'Std_Approx_DeltaG_kcal_per_mol': std_approx,
            'Mean_Prodigy_DeltaG_kcal_per_mol': mean_prodigy,
            'Std_Prodigy_DeltaG_kcal_per_mol': std_prodigy,
            'Mean_Contacts': mean_contacts,
            'Std_Contacts': std_contacts
        })

    return pd.DataFrame(summary_stats)

def create_comprehensive_visualizations(all_results: Dict, stat_results: Dict, config: Config):
    """
    Create comprehensive visualizations for all methods.
    """
    print("\nCreating comprehensive visualizations...")

    # Create a figure with subplots for all methods
    n_methods = len([k for k in all_results.keys() if k in ['mse', 'persistence', 'ensemble']])

    if n_methods > 0:
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 8))
        if n_methods == 1:
            axes = [axes]

        idx = 0

        # MSE heatmap
        if 'mse' in all_results:
            df = all_results['mse']
            matrix = df.pivot_table(
                index='Binding_Entity',
                columns='Antigen',
                values='Center_Position_MSE'
            )

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.2f',
                cmap='viridis_r',
                ax=axes[idx],
                cbar_kws={'label': 'MSE (Å²)'}
            )
            axes[idx].set_title(f'Binding Pose Consistency\n(p={stat_results.get("mse", {}).get("p_value", 1):.3f})')
            idx += 1

        # Persistence heatmap
        if 'persistence' in all_results:
            df = all_results['persistence']
            matrix = df.pivot_table(
                index='Binding_Entity',
                columns='Antigen',
                values='Mean_Persistence'
            )

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                ax=axes[idx],
                cbar_kws={'label': 'Persistence Score'}
            )
            axes[idx].set_title(
                f'Interface Persistence\n(p={stat_results.get("persistence", {}).get("p_value", 1):.3f})')
            idx += 1

        # Ensemble score heatmap
        if 'ensemble' in all_results:
            df = all_results['ensemble']
            matrix = df.pivot_table(
                index='Binding_Entity',
                columns='Antigen',
                values='Ensemble_Score'
            )

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                ax=axes[idx],
                cbar_kws={'label': 'Ensemble Score'}
            )
            axes[idx].set_title(f'Ensemble Score\n(p={stat_results.get("ensemble", {}).get("p_value", 1):.3f})')

        plt.tight_layout()
        plt.savefig(config.results_dir / "comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Comprehensive visualization saved to: {config.results_dir / 'comprehensive_analysis.png'}")


def save_comprehensive_report(all_results: Dict, stat_results: Dict, config: Config):
    """
    Save a comprehensive report combining all results.
    """
    # Create a markdown report
    report_path = config.results_dir / "comprehensive_report.md"

    with open(report_path, 'w') as f:
        f.write("# AlphaFold3 Nanobody Evaluation - Comprehensive Report\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")

        for method, stats in stat_results.items():
            f.write(f"### {method.upper()}\n")
            f.write(f"- Diagonal mean: {stats['diagonal_mean']:.3f} ± {stats['diagonal_std']:.3f}\n")
            f.write(f"- Off-diagonal mean: {stats['off_diagonal_mean']:.3f} ± {stats['off_diagonal_std']:.3f}\n")
            f.write(f"- P-value: {stats['p_value']:.4f}\n")
            f.write(f"- Effect size (Cohen's d): {stats['effect_size']:.3f}\n\n")

        # Top predictions by ensemble score
        if 'ensemble' in all_results:
            f.write("## Top Predictions by Ensemble Score\n\n")
            ensemble_df = all_results['ensemble'].sort_values('Ensemble_Score', ascending=False)

            f.write("| Rank | Binding Entity | Antigen | Ensemble Score |\n")
            f.write("|------|---------------|---------|----------------|\n")

            for i, row in ensemble_df.head(10).iterrows():
                f.write(f"| {i + 1} | {row['Binding_Entity']} | {row['Antigen']} | {row['Ensemble_Score']:.3f} |\n")

        f.write("\n## Files Generated\n\n")
        f.write("- MSE results: `center_position_mse_results.csv`\n")
        f.write("- Binding energy results: `binding_energy_results.csv`\n")
        f.write("- Interface persistence: `interface_persistence_results.csv`\n")
        f.write("- Confidence metrics: `confidence_metrics_results.csv`\n")
        f.write("- Ensemble scores: `ensemble_scores.csv`\n")
        f.write("- Statistical analysis: `statistical_analysis.csv`\n")

    print(f"\nComprehensive report saved to: {report_path}")


def cleanup_intermediate_files(config: Config):
    """
    Clean up intermediate files after analysis.

    Args:
        config: Configuration object
    """
    print("Cleaning up intermediate files...")

    # Remove temporary directories
    temp_dirs = [
        config.temp_dir / "prodigy_pdbs",
        config.model_dir
    ]

    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("  Cleanup completed")


def extract_models_af3_zip(input_dir, config):
    """
    Extract models from AlphaFold3 ZIP files (all 5 models per seed).

    Args:
        input_dir: Directory containing ZIP files
        config: Configuration object

    Returns:
        Dictionary mapping combinations to lists of model paths
    """
    print("Extracting models from AlphaFold3 ZIP files...")

    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    if not zip_files:
        raise ValueError("No ZIP files found in " + str(input_dir))

    # Handle special case filenames
    renamed_files = {}
    for f in zip_files:
        if "seed_4B" in f:
            new_f = f.replace("seed_4B", "seed_4b")
            os.rename(f, new_f)
            renamed_files[new_f] = f

    # Reload zip files after renaming
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))

    # Group ZIP files by combination
    combination_to_zips = {}
    for zip_file in zip_files:
        basename = os.path.basename(zip_file)
        parts = basename.split('_')

        # Extract combination info
        if len(parts) >= 4:
            binding_entity = parts[1] + '_' + parts[2]
            antigen = parts[3]
            combination = binding_entity + "_" + antigen

            if combination not in combination_to_zips:
                combination_to_zips[combination] = []

            combination_to_zips[combination].append(zip_file)

    # Extract all models for each combination
    combination_to_models = {}

    for combination, zip_list in combination_to_zips.items():
        print("  Processing " + combination + " (" + str(len(zip_list)) + " seeds)")

        if len(zip_list) < config.analysis_config["min_seeds"]:
            print("  Warning: " + combination + " has only " + str(len(zip_list)) +
                  " seeds, minimum required is " + str(config.analysis_config["min_seeds"]))
            continue

        model_paths = []
        for zip_path in zip_list:
            # Extract all 5 models instead of just the best one
            extracted_models = extract_all_models_from_zip(zip_path, config.model_dir)
            model_paths.extend(extracted_models)

        if model_paths:
            combination_to_models[combination] = model_paths

    # Restore original filenames
    for new_f, orig_f in renamed_files.items():
        os.rename(new_f, orig_f)

    return combination_to_models