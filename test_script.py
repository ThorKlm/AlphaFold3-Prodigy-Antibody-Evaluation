# test_prodigy_with_sample.py
import os
import glob
import zipfile
import tempfile
import subprocess
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO


def create_dir_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def extract_best_model_from_zip(zip_path, output_dir):
    """Extract the best model (first CIF file) from a ZIP archive."""
    print(f"Extracting from ZIP: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        cif_files = [f for f in zf.namelist() if f.endswith('.cif')]

        if not cif_files:
            print(f"No CIF files found in {zip_path}")
            return None

        # Select the best model (first in sorted order)
        best_model = sorted(cif_files)[0]
        print(f"Best model file: {best_model}")

        # Generate output filename
        output_filename = f"{Path(zip_path).stem}_best_model.cif"
        output_path = os.path.join(output_dir, output_filename)

        # Extract the file
        with open(output_path, 'wb') as out_f:
            out_f.write(zf.read(best_model))

        print(f"Extracted to: {output_path}")
        return output_path


def convert_cif_to_pdb(cif_path, output_dir):
    """Convert CIF file to PDB format."""
    print(f"Converting CIF to PDB: {cif_path}")

    # Parse the CIF file
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_path)

    # Generate output filename
    output_filename = f"{Path(cif_path).stem}.pdb"
    output_path = os.path.join(output_dir, output_filename)

    # Write structure to PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)

    print(f"Converted to PDB: {output_path}")
    return output_path


def run_prodigy_on_pdb(pdb_path, temp=25.0):
    """Run PRODIGY on a PDB file."""
    print(f"Running PRODIGY on: {pdb_path}")

    try:
        # Run PRODIGY without selection arguments
        cmd = ["prodigy", f"--temperature={temp}", pdb_path]

        # Set UTF-8 encoding for subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        print(f"Running command: {' '.join(cmd)}")

        # Run with full error capture
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        print(f"Return code: {result.returncode}")

        if result.stdout:
            print(f"Output:\n{result.stdout}")

        if result.stderr:
            print(f"Error:\n{result.stderr}")

        # Parse output for binding energy
        binding_energy = None
        for line in result.stdout.split('\n'):
            if "Predicted binding affinity" in line:
                print(f"Found binding energy line: {line}")
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        binding_energy = float(parts[1].strip().split()[0])
                        print(f"Extracted binding energy: {binding_energy}")
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing binding energy: {e}")

        return binding_energy

    except Exception as e:
        print(f"Error running PRODIGY: {e}")
        return None


def main():
    """Main function to process ZIP file and run PRODIGY."""
    # Get base directory from user
    base_dir = input("Enter path to your AlphaFold3Eval directory: ")

    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Look for alphafold_prediction_samples directory
    samples_dir = os.path.join(base_dir, "alphafold_prediction_samples")
    if not os.path.exists(samples_dir):
        # Try to find ZIP files in the base directory
        samples_dir = base_dir

    # Find ZIP files
    zip_files = glob.glob(os.path.join(samples_dir, "*.zip"))

    if not zip_files:
        print(f"No ZIP files found in {samples_dir}")
        return

    # Sort ZIP files and select the first one
    zip_files.sort()
    first_zip = zip_files[0]
    print(f"Selected ZIP file: {first_zip}")

    # Create temp_structures directory
    temp_structures_dir = os.path.join(base_dir, "temp_structures")
    create_dir_if_not_exists(temp_structures_dir)

    # Extract best model from ZIP
    cif_path = extract_best_model_from_zip(first_zip, temp_structures_dir)

    if not cif_path:
        print("Failed to extract model from ZIP")
        return

    # Convert CIF to PDB
    pdb_path = convert_cif_to_pdb(cif_path, temp_structures_dir)

    # Run PRODIGY on the PDB file
    binding_energy = run_prodigy_on_pdb(pdb_path)

    if binding_energy is not None:
        print(f"\nSuccess! Binding energy: {binding_energy} kcal/mol")
    else:
        print("\nFailed to calculate binding energy")


if __name__ == "__main__":
    main()