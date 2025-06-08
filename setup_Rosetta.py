# Open WSL terminal
'''
# Step 1.1: Create the Rosetta conda environment
conda create -n Rosetta python=3.8 -y
conda activate Rosetta

# Step 1.2: Install required build and runtime dependencies
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    unzip \
    libboost-all-dev \
    libz-dev \
    libbz2-dev \
    liblzma-dev \
    libfftw3-dev \
    libx11-dev \
    libopenmpi-dev \
    openmpi-bin

# Step 1.3 (Optional): Install PyRosetta dependencies
conda install -c conda-forge numpy pandas biopython
'''

'''
# Make a directory for Rosetta and enter it
mkdir -p ~/rosetta && cd ~/rosetta

# Use wget to download the source archive
wget https://downloads.rosettacommons.org/downloads/academic/3.14/rosetta_src_3.14_bundle.tar.bz2

# Unpack the archive
tar -xjf rosetta_src_2023.08.61115_bundle.tar.bz2
'''

'''
(abnativ_env) thor@Thor03:~/abnativ$ python abnativ_script.py score -i /mnt/c/WSL/sequences/nanobodies.fasta -odir /mnt/c/WSL/results -oid mybatch -align -ncpu 4 -v
'''
# Verification with
# http://research.naturalantibody.com/nbsequencesearchinput
'''
python abnativ/abnativ_script.py score -i /mnt/c/WSL/sequences/test_set_nanobodies.fasta -odir /mnt/c/WSL/results -oid test_set_nanobodies -align -ncpu 4 -v
'''