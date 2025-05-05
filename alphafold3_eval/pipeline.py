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

from alphafold3_eval.config import Config
from alphafold3_eval.structure_uitls import (
    extract_best_model_from_zip,
    parse_model_filename
)
from alphafold3_eval.alignment import (
    align_structure_to_reference,
    compute_antigen_aligned_mse
)
from alphafold3_eval.binding_analysis import (
    calculate_binding_energies,
    calculate_interface_rmsd,
    extract_interface_centroid,
    detect_antibody_type
)
from alphafold3_eval.visualization import (
    plot_mse_distributions,
    plot_binding_energy_boxplot,
    plot_energy_vs_mse,
    create_confusion_matrix_heatmap,
    create_combined_plot,
    create_matrix_visualization
)


def extract_models(input_dir: str, config: Config) -> Dict[str, List[str]]:
    """
    Extract best models from all ZIP files in the input directory.

    Args:
        input_dir: Directory containing ZIP files
        config: Configuration object

    Returns:
        Dictionary mapping combinations to lists of model paths
    """
    print("Extracting best models from ZIP files...")

    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))
    if not zip_files:
        raise ValueError(f"No ZIP files found in {input_dir}")

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
            combination = f"{binding_entity}_{antigen}"

            if combination not in combination_to_zips:
                combination_to_zips[combination] = []

            combination_to_zips[combination].append(zip_file)

    # Extract best models for each combination
    combination_to_models = {}

    for combination, zip_list in combination_to_zips.items():
        print(f"  Processing {combination} ({len(zip_list)} seeds)")

        if len(zip_list) < config.analysis_config["min_seeds"]:
            print(f"Warning: {combination} has only {len(zip_list)} seeds, minimum required is {config.analysis_config['min_seeds']}")
            continue

        model_paths = []
        for zip_path in zip_list:
            # Extract model
            output_filename = f"{Path(zip_path).stem}_model_0.cif"
            model_path = extract_best_model_from_zip(
                zip_path,
                config.model_dir,
                output_filename
            )

            if model_path:
                model_paths.append(model_path)

        if model_paths:
            combination_to_models[combination] = model_paths

    # Restore original filenames
    for new_f, orig_f in renamed_files.items():
        os.rename(new_f, orig_f)

    return combination_to_models


def analyze_mse(combination_to_models: Dict[str, List[str]], config: Config) -> pd.DataFrame:
    """
    Analyze antigen-aligned MSE for each combination.

    Args:
        combination_to_models: Dictionary mapping combinations to lists of model paths
        config: Configuration object

    Returns:
        DataFrame with MSE results
    """
    print("Analyzing binding pose consistency (MSE)...")

    results = []

    for combination, model_paths in combination_to_models.items():
        print(f"  Computing MSE for {combination}")

        # Parse combination
        parts = combination.split('_')
        binding_entity = parts[0] + '_' + parts[1]
        antigen = parts[2]

        # Compute MSE
        mse = compute_antigen_aligned_mse(model_paths)

        # Detect antibody type
        antibody_type = detect_antibody_type(binding_entity)

        results.append({
            'Binding_Entity': binding_entity,
            'Antigen': antigen,
            'Combination': combination,
            'Center_Position_MSE': mse,
            'Num_Seeds': len(model_paths),
            'Antibody_Type': antibody_type
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(config.output_files["mse_results"], index=False)
    print(f"MSE results saved to: {config.output_files['mse_results']}")

    return results_df


def analyze_binding_energy(combination_to_models: Dict[str, List[str]],
                         config: Config, use_prodigy: bool = True) -> pd.DataFrame:
    """
    Analyze binding energy for each model using both methods.

    Args:
        combination_to_models: Dictionary mapping combinations to lists of model paths
        config: Configuration object
        use_prodigy: Whether to use PRODIGY for binding energy calculation

    Returns:
        DataFrame with binding energy results
    """
    print("Analyzing binding energy...")

    parser = MMCIFParser(QUIET=True)

    # Create temp directory for PRODIGY PDB files
    prodigy_temp_dir = os.path.join(config.temp_dir, "prodigy_pdbs")
    os.makedirs(prodigy_temp_dir, exist_ok=True)

    records = []

    for combination, model_paths in combination_to_models.items():
        print(f"  Computing binding energy for {combination}")

        # Parse combination
        parts = combination.split('_')
        binding_entity = parts[0] + '_' + parts[1]
        antigen = parts[2]

        # Create reference for alignment
        reference_structure = None

        # Process each model
        for model_path in model_paths:
            try:
                # Parse metadata
                meta = parse_model_filename(model_path)
                if not meta:
                    continue

                # Load structure
                structure = parser.get_structure(combination, model_path)

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
                print(f"Error processing {model_path}: {e}")

    # Create DataFrame
    df_models = pd.DataFrame(records)

    if df_models.empty:
        raise ValueError("No valid models processed - check CIF integrity or parsing.")

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
    print(f"Binding energy results saved to: {config.output_files['binding_energy']}")

    # Save scaled results
    scaled_path = str(config.output_files["binding_energy"]).replace(".csv", "_scaled.csv")
    df_save_scaled.to_csv(scaled_path, index=False)
    print(f"Scaled binding energy results saved to: {scaled_path}")

    # Clean up temporary PDB files
    shutil.rmtree(prodigy_temp_dir, ignore_errors=True)

    return df_models, df_models_scaled


def create_summary_dataframe(df_models: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary DataFrame with mean values for each combination.

    Args:
        df_models: DataFrame with per-model results

    Returns:
        DataFrame with summary statistics
    """
    summary = []

    for combo, group in df_models.groupby("Combination"):
        # Calculate RMSD of interface centroids
        centroids = np.stack(group["Interface_Centroid"].values)
        from scipy.spatial.distance import pdist, squareform
        rmsd_matrix = squareform(pdist(centroids, metric='euclidean'))
        rmsd = np.mean(rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)])

        # Calculate mean binding energies
        mean_approx_energy = group["Approx_DeltaG_kcal_per_mol"].mean()

        # Calculate mean PRODIGY energy if available
        prodigy_values = group["Prodigy_DeltaG_kcal_per_mol"].dropna()
        mean_prodigy_energy = prodigy_values.mean() if not prodigy_values.empty else np.nan

        # Get other information
        binding_entity = group["Binding_Entity"].iloc[0]
        antigen = group["Antigen"].iloc[0]
        antibody_type = group["Antibody_Type"].iloc[0]
        is_scaled = group["Is_Scaled"].iloc[0]

        # Add summary record
        summary.append({
            "Combination": combo,
            "Binding_Entity": binding_entity,
            "Antigen": antigen,
            "Mean_Approx_DeltaG_kcal_per_mol": mean_approx_energy,
            "Mean_Prodigy_DeltaG_kcal_per_mol": mean_prodigy_energy,
            "Interface_Deviation_RMSD": rmsd,
            "Num_Seeds": len(group),
            "Antibody_Type": antibody_type,
            "Is_Scaled": is_scaled
        })

    return pd.DataFrame(summary)


def create_visualizations(df_models: pd.DataFrame,
                          df_summary: pd.DataFrame,
                          df_mse: pd.DataFrame,
                          config: Config,
                          is_scaled: bool = False) -> None:
    """
    Create visualizations from the analysis results.

    Args:
        df_models: DataFrame with per-model results
        df_summary: DataFrame with summary statistics
        df_mse: DataFrame with MSE results
        config: Configuration object
        is_scaled: Whether the data is scaled (for output filenames)
    """
    suffix = "_scaled" if is_scaled else ""
    scaling_text = " (Two-Chain Antibodies Scaled by 0.5)" if is_scaled else ""

    print(f"Creating visualizations{scaling_text}...")

    # Plot 1: MSE distributions
    plot_mse_distributions(
        df_mse,
        output_path=str(config.plot_files["mse_plot"]).replace(".png", f"{suffix}.png"),
        x_col="Combination",
        y_col="Center_Position_MSE",
        title=f"Binding Pose Consistency Across AlphaFold3 Seeds{scaling_text}"
    )
    print(f"MSE plot saved to: {str(config.plot_files['mse_plot']).replace('.png', f'{suffix}.png')}")

    # Plot 2: Approximate Binding Energy boxplot
    plot_binding_energy_boxplot(
        df_models,
        output_path=str(config.plot_files["energy_plot"]).replace(".png", f"_approx{suffix}.png"),
        x_col="Combination",
        y_col="Approx_DeltaG_kcal_per_mol",
        title=f"Approximated Binding Energy Across Seeds{scaling_text}"
    )
    print(
        f"Approximated binding energy plot saved to: {str(config.plot_files['energy_plot']).replace('.png', f'_approx{suffix}.png')}")

    # Plot 3: PRODIGY Binding Energy boxplot
    plot_binding_energy_boxplot(
        df_models,
        output_path=str(config.plot_files["energy_plot"]).replace(".png", f"_prodigy{suffix}.png"),
        x_col="Combination",
        y_col="Prodigy_DeltaG_kcal_per_mol",
        title=f"PRODIGY Binding Energy Across Seeds{scaling_text}"
    )
    print(
        f"PRODIGY binding energy plot saved to: {str(config.plot_files['energy_plot']).replace('.png', f'_prodigy{suffix}.png')}")

    # Plot 4: Energy vs RMSD scatter plot (approximate)
    plot_energy_vs_mse(
        df_summary,
        output_path=str(config.plot_files["combined_plot"]).replace(".png", f"_approx{suffix}.png"),
        x_col="Interface_Deviation_RMSD",
        y_col="Mean_Approx_DeltaG_kcal_per_mol",
        combo_col="Combination",
        title=f"Mean Approximated ΔG vs Mean Pose Deviation per Combination{scaling_text}",
        show_regression=False
    )
    print(
        f"MSE vs approximate energy plot saved to: {str(config.plot_files['combined_plot']).replace('.png', f'_approx{suffix}.png')}")

    # Plot 5: Energy vs RMSD scatter plot (PRODIGY)
    plot_energy_vs_mse(
        df_summary,
        output_path=str(config.plot_files["combined_plot"]).replace(".png", f"_prodigy{suffix}.png"),
        x_col="Interface_Deviation_RMSD",
        y_col="Mean_Prodigy_DeltaG_kcal_per_mol",
        combo_col="Combination",
        title=f"Mean PRODIGY ΔG vs Mean Pose Deviation per Combination{scaling_text}",
        show_regression=False
    )
    print(
        f"MSE vs PRODIGY energy plot saved to: {str(config.plot_files['combined_plot']).replace('.png', f'_prodigy{suffix}.png')}")

    # Plot 6: Confusion matrix heatmap (approximate)
    create_confusion_matrix_heatmap(
        df_summary,
        output_path=str(config.results_dir / f"approx_binding_energy_matrix{suffix}.png"),
        binding_col="Binding_Entity",
        antigen_col="Antigen",
        value_col="Mean_Approx_DeltaG_kcal_per_mol",
        title=f"Approximated Binding Energy Confusion Matrix{scaling_text}"
    )
    print(
        f"Approximate energy confusion matrix saved to: {config.results_dir / f'approx_binding_energy_matrix{suffix}.png'}")

    # Plot 7: Confusion matrix heatmap (PRODIGY)
    create_confusion_matrix_heatmap(
        df_summary,
        output_path=str(config.results_dir / f"prodigy_binding_energy_matrix{suffix}.png"),
        binding_col="Binding_Entity",
        antigen_col="Antigen",
        value_col="Mean_Prodigy_DeltaG_kcal_per_mol",
        title=f"PRODIGY Binding Energy Confusion Matrix{scaling_text}"
    )
    print(
        f"PRODIGY energy confusion matrix saved to: {config.results_dir / f'prodigy_binding_energy_matrix{suffix}.png'}")

    # Plot 8: Matrix visualization
    create_matrix_visualization(
        df_summary,
        df_mse,
        output_path=str(config.results_dir / f"matrix_visualization{suffix}.png"),
        scale_two_chain=is_scaled,
        title=f"AlphaFold3 Binding Evaluation Matrix{scaling_text}"
    )
    print(f"Matrix visualization saved to: {config.results_dir / f'matrix_visualization{suffix}.png'}")


def cleanup_intermediate_files(config: Config) -> None:
    """
    Clean up intermediate files after analysis.

    Args:
        config: Configuration object
    """
    print("Cleaning up intermediate files...")

    # Remove model directory
    if os.path.exists(config.model_dir):
        shutil.rmtree(config.model_dir)
        print(f"Removed model directory: {config.model_dir}")

    # Remove temp directory
    if os.path.exists(config.temp_dir):
        shutil.rmtree(config.temp_dir)
        print(f"Removed temp directory: {config.temp_dir}")


def run_pipeline(input_dir: str, config: Config, use_prodigy: bool = True,
                 clean_intermediate: bool = False) -> None:
    """
    Run the complete analysis pipeline.

    Args:
        input_dir: Directory containing input ZIP files
        config: Configuration object
        use_prodigy: Whether to use PRODIGY for binding energy calculation
        clean_intermediate: Whether to clean up intermediate files after analysis
    """
    print(f"Starting AlphaFold3 binding evaluation pipeline...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {config.results_dir}")
    print(f"Using PRODIGY: {use_prodigy}")

    # Step 1: Extract models
    combination_to_models = extract_models(input_dir, config)

    if not combination_to_models:
        raise ValueError("No valid models extracted")

    # Step 2: Analyze MSE
    df_mse = analyze_mse(combination_to_models, config)

    # Step 3: Analyze binding energy (both unscaled and scaled)
    df_models, df_models_scaled = analyze_binding_energy(combination_to_models, config, use_prodigy)

    # Step 4: Create summary dataframes
    df_summary = create_summary_dataframe(df_models)
    df_summary_scaled = create_summary_dataframe(df_models_scaled)

    # Step 5: Save combined results (unscaled)
    df_combined = pd.merge(
        df_summary,
        df_mse[["Combination", "Center_Position_MSE"]],
        on="Combination",
        how="outer"
    )
    df_combined.to_csv(config.output_files["combined_results"], index=False)
    print(f"Combined results saved to: {config.output_files['combined_results']}")

    # Step 6: Save combined results (scaled)
    df_combined_scaled = pd.merge(
        df_summary_scaled,
        df_mse[["Combination", "Center_Position_MSE"]],
        on="Combination",
        how="outer"
    )
    scaled_path = str(config.output_files["combined_results"]).replace(".csv", "_scaled.csv")
    df_combined_scaled.to_csv(scaled_path, index=False)
    print(f"Scaled combined results saved to: {scaled_path}")

    # Step 7: Create visualizations (unscaled)
    create_visualizations(df_models, df_summary, df_mse, config, is_scaled=False)

    # Step 8: Create visualizations (scaled)
    create_visualizations(df_models_scaled, df_summary_scaled, df_mse, config, is_scaled=True)

    # Step 9: Clean up intermediate files if requested
    if clean_intermediate:
        cleanup_intermediate_files(config)

    print("Pipeline completed successfully!")