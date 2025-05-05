"""
Command-line interface for AlphaFold3 binding evaluation.
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from alphafold3_eval.config import Config
from alphafold3_eval.structure_uitls import extract_best_model_from_zip, parse_model_filename
from alphafold3_eval.binding_analysis import run_prodigy_on_pdb

# Import pipeline module separately
import alphafold3_eval.pipeline as pipeline


def check_prodigy_installation():
    """Check if PRODIGY is installed and working."""
    import tempfile
    import subprocess

    print("Checking PRODIGY installation...")

    try:
        # Create a simple test command
        cmd = ["prodigy", "--help"]

        # Run with env var for encoding
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            print("PRODIGY installation check: OK")
            return True
        else:
            print(f"PRODIGY command-line tool check failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"PRODIGY installation check failed: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AlphaFold3 Binding Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Create a group for input requirements - only required if not using --check
    input_group = parser.add_argument_group('Input/Output')
    input_group.add_argument(
        "--input", "-i",
        help="Directory containing input ZIP files with AlphaFold3 predictions"
    )

    input_group.add_argument(
        "--output", "-o", default=None,
        help="Output directory for results and plots"
    )

    # Parameters group
    params_group = parser.add_argument_group('Analysis Parameters')
    params_group.add_argument(
        "--temp", "-t", type=float, default=25.0,
        help="Temperature in Celsius for PRODIGY binding energy calculation"
    )

    params_group.add_argument(
        "--cutoff", "-c", type=float, default=5.0,
        help="Distance cutoff in Angstroms for interface detection"
    )

    params_group.add_argument(
        "--min-seeds", "-m", type=int, default=5,
        help="Minimum number of seeds required for a valid analysis"
    )

    # Flags group
    flags_group = parser.add_argument_group('Flags')
    flags_group.add_argument(
        "--skip-prodigy", action="store_true",
        help="Skip PRODIGY binding energy calculation and use only simple contact-based model"
    )

    flags_group.add_argument(
        "--no-scaling", action="store_true",
        help="Disable scaling binding energies of two-chain antibodies"
    )

    flags_group.add_argument(
        "--clean", action="store_true",
        help="Clean intermediate files after analysis"
    )

    flags_group.add_argument(
        "--check", action="store_true",
        help="Run a check to verify PRODIGY installation"
    )

    args = parser.parse_args()

    # Validate arguments - input is required unless just checking PRODIGY
    if not args.check and not args.input:
        parser.error("the following arguments are required: --input/-i")

    return args


def test_prodigy_with_sample_pdb(config: Config):
    """
    Test PRODIGY with a sample PDB file to verify it's working.

    Args:
        config: Configuration object
    """
    import tempfile
    import urllib.request

    print("Testing PRODIGY with a sample PDB file...")

    # Create temp directory if it doesn't exist
    os.makedirs(config.temp_dir, exist_ok=True)

    # Download a sample PDB file
    sample_pdb_path = os.path.join(config.temp_dir, "sample.pdb")
    try:
        urllib.request.urlretrieve(
            "https://files.rcsb.org/download/1AY1.pdb",
            sample_pdb_path
        )
        print(f"Downloaded sample PDB: {sample_pdb_path}")
    except Exception as e:
        print(f"Failed to download sample PDB: {e}")
        return False

    # Run PRODIGY on the sample PDB
    try:
        binding_energy = run_prodigy_on_pdb(
            sample_pdb_path,
            temp=config.prodigy_config["temperature"]
        )

        if binding_energy is not None:
            print(f"PRODIGY test successful: Binding energy = {binding_energy} kcal/mol")
            return True
        else:
            print("PRODIGY test failed: No binding energy returned")
            return False

    except Exception as e:
        print(f"PRODIGY test failed: {e}")
        return False

    finally:
        # Clean up the sample PDB
        if os.path.exists(sample_pdb_path):
            os.remove(sample_pdb_path)


def run_pipeline_wrapper(input_dir: str, config: Config, use_prodigy: bool = True, clean_intermediate: bool = False) -> None:
    """
    Wrapper for the pipeline's run_pipeline function to handle import issues.

    Args:
        input_dir: Directory containing input ZIP files
        config: Configuration object
        use_prodigy: Whether to use PRODIGY for binding energy calculation
        clean_intermediate: Whether to clean up intermediate files after analysis
    """
    # Get all functions from the pipeline module
    for attr_name in dir(pipeline):
        if attr_name.startswith('__'):
            continue

        attr = getattr(pipeline, attr_name)
        if callable(attr) and attr_name == 'run_pipeline':
            # Call the function directly
            attr(input_dir, config, use_prodigy, clean_intermediate)
            return

    # If we get here, the function wasn't found
    raise AttributeError(f"Could not find run_pipeline function in {pipeline.__file__}")


def main() -> None:
    """Main entry point for the command-line tool."""
    print("=" * 80)
    print("AlphaFold3 Binding Evaluation Tool")
    print("=" * 80)

    args = parse_args()

    # Set up configuration with output directory from args or current directory
    output_dir = args.output if args.output else os.getcwd()
    config = Config(base_dir=output_dir)

    # Update configuration with command-line arguments
    config.prodigy_config["temperature"] = args.temp
    config.analysis_config["contact_cutoff"] = args.cutoff
    config.analysis_config["min_seeds"] = args.min_seeds
    config.analysis_config["use_prodigy"] = not args.skip_prodigy
    config.analysis_config["scale_two_chain"] = not args.no_scaling

    # Run PRODIGY check if requested
    if args.check:
        prodigy_installed = check_prodigy_installation()
        if prodigy_installed and not args.skip_prodigy:
            test_result = test_prodigy_with_sample_pdb(config)
            if not test_result:
                print("\nPRODIGY test failed. You can:")
                print("  1. Fix PRODIGY installation issues (recommended)")
                print("  2. Use the --skip-prodigy flag to use simple contact-based model")
                sys.exit(1)

        print("\nPRODIGY check completed.")
        if not args.input:
            return  # Exit if only doing a check

    # Set environment variables for Python I/O encoding
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Run the analysis pipeline
    try:
        run_pipeline_wrapper(
            input_dir=args.input,
            config=config,
            use_prodigy=not args.skip_prodigy,
            clean_intermediate=args.clean
        )

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()