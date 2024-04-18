#!/bin/bash

TARGET_DIR="Torgo"

# Check if zip files are already present
if [ ! -f "${TARGET_DIR}/F.tar.bz2" ] || [ ! -f "${TARGET_DIR}/FC.tar.bz2" ] || [ ! -f "${TARGET_DIR}/M.tar.bz2" ] || [ ! -f "${TARGET_DIR}/MC.tar.bz2" ]; then
    # If any zip file is not present, download them
    wget http://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2 -P Torgo/
    wget http://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2 -P Torgo/
    wget http://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2 -P Torgo/
    wget http://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2 -P Torgo/
fi

# Extract files
tar -xvf "${TARGET_DIR}/F.tar.bz2" -C "${TARGET_DIR}/dysarthria"
tar -xvf "${TARGET_DIR}/FC.tar.bz2" -C "${TARGET_DIR}/non_dysarthria"
tar -xvf "${TARGET_DIR}/M.tar.bz2" -C "${TARGET_DIR}/dysarthria"
tar -xvf "${TARGET_DIR}/MC.tar.bz2" -C "${TARGET_DIR}/non_dysarthria"
