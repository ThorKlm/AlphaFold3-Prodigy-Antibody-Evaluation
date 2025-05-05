# Setting up PRODIGY on Windows

This guide provides detailed instructions for installing and configuring PRODIGY (PROtein binDIng enerGY prediction) on Windows for use with the AlphaFold3 binding evaluation project.

## Prerequisites

Before installing PRODIGY, ensure you have:

- Windows 10 or 11
- Python 3.8+ (Python 3.9 recommended)
- Git for Windows
- A properly set up virtual environment for the project

## Installation Steps

### 1. Install Required Tools

First, activate your project's virtual environment and install the necessary dependencies:

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Install required packages
pip install requests beautifulsoup4 numpy scipy networkx
```

### 2. Clone the PRODIGY Repository

```powershell
# Create a directory for external tools
mkdir external_tools
cd external_tools

# Clone the PRODIGY repository
git clone https://github.com/haddocking/prodigy.git
cd prodigy
```

### 3. Install PRODIGY in Development Mode

```powershell
# Install in development mode
pip install -e .
```

### 4. Configure PRODIGY for Windows

PRODIGY was primarily developed for Unix-like systems, so we need to make a few Windows-specific adjustments:

#### 4.1. Fix Path Handling in Scripts

Some PRODIGY scripts may use Unix-style path handling. To fix this, edit the following files:

**prodigy/predict_IC.py:**

```python
# Find and modify any lines that use os.path.join but assume Unix-style separators
# Example:
# Before: output_file = os.path.join(output_dir, pdb_id + "/intf_contacts.list")
# After: output_file = os.path.join(output_dir, pdb_id, "intf_contacts.list")
```

#### 4.2. Fix Temporary File Handling

PRODIGY sometimes assumes Unix-style temporary directory handling. Edit any files that use `/tmp` paths:

```python
# Before: tmp_dir = '/tmp'
# After: tmp_dir = tempfile.gettempdir()
```

Make sure to add `import tempfile` at the top of any files where you make this change.

### 5. Test PRODIGY Installation

To verify that PRODIGY was installed correctly, run:

```powershell
# Change back to the project root directory
cd ..\..

# Test PRODIGY import
python -c "import prodigy; print('PRODIGY imported successfully!')"
```

You should see the message "PRODIGY imported successfully!" if the installation worked.

## Using PRODIGY via Command Line

PRODIGY also provides a command-line interface. To test it:

```powershell
# Test the PRODIGY command-line tool
prodigy --help
```

If everything is set up correctly, you should see the help message for the PRODIGY command-line tool.

## Troubleshooting

### ImportError or ModuleNotFoundError

If you get an error about missing modules, install the required dependencies:

```powershell
pip install numpy scipy networkx
```

### Path-related Errors

Windows uses backslashes (`\`) for paths, while PRODIGY may expect forward slashes (`/`). If you encounter path-related errors, consider using raw strings for Windows paths:

```python
# Use raw strings for Windows paths
path = r"C:\path\to\file"
```

### Permission Issues

If you encounter permission issues when PRODIGY tries to write temporary files:

1. Ensure you're running Python with appropriate permissions
2. Configure PRODIGY to use a different temporary directory:

```python
import os
import tempfile

# Set a custom temp directory in your user space
os.environ['TMPDIR'] = os.path.join(tempfile.gettempdir(), 'prodigy_temp')
os.makedirs(os.environ['TMPDIR'], exist_ok=True)
```

## Integration with AlphaFold3 Binding Evaluation

The AlphaFold3 binding evaluation project is designed to work with PRODIGY in two ways:

1. Using the Python API (preferred method)
2. Using the command-line interface (fallback method)

Both methods are implemented in `binding_analysis.py`, and the tool will automatically try the Python API first and fall back to the command-line interface if needed.

## Manual PRODIGY Analysis

To manually analyze a structure with PRODIGY:

```powershell
# Using the Python API
python -c "from prodigy.predict_IC import predict_ic; from prodigy.lib.parsers import parse_structure; struct = parse_structure('path/to/structure.pdb'); result = predict_ic(struct, ['A'], ['B']); print(result)"

# Using the command-line tool
prodigy --selection=A B structure.pdb
```

Replace `'A'` and `'B'` with the actual chain IDs of your structure.