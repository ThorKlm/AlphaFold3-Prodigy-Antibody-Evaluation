#!/usr/bin/env python
"""
Example script for running the AlphaFold3 binding evaluation pipeline.
"""
import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphafold3_eval.config import Config
from alphafold3_eval import pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AlphaFold3 Binding Evaluation Example Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory containing input ZIP files with AlphaFold3 predictions"
    )

    parser.add_argument(
        "--output", "-o", default="./results",
        help="Output directory for results and plots"
    )

    parser.add_argument(
        "--temp", "-t", type=float, default=25.0,
        help="Temperature in Celsius for PRODIGY binding energy calculation"
    )

    parser.add_argument(
        "--cutoff", "-c", type=float, default=5.0,
        help="Distance cutoff in Angstroms for interface detection"
    )

    parser.add_argument(
        "--skip-prodigy", action="store_true",
        help="Skip PRODIGY binding energy calculation and use simple contact-based model"
    )

    parser.add_argument(
        "--clean", action="store_true",
        help="Clean intermediate files after analysis"
    )

    return parser.parse_args()


def main():
    """Main function."""
    print("=" * 80)
    print("AlphaFold3 Binding Evaluation Example Script")
    print("=" * 80)

    # Parse arguments
    args = parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Configure the analysis
    config = Config(base_dir=args.output)
    config.prodigy_config["temperature"] = args.temp
    config.analysis_config["contact_cutoff"] = args.cutoff

    # Run the pipeline
    try:
        pipeline.run_pipeline(
            input_dir=args.input,
            config=config,
            use_prodigy=not args.skip_prodigy,
            clean_intermediate=args.clean
        )
        print("\n✅ Analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()