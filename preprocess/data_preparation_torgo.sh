#!/bin/bash

# Define the base URL and target directory
BASE_URL="http://www.cs.toronto.edu/~complingweb/data/TORGO/"
TARGET_DIR="Torgo"

# Define the file names
FILES=(
    "F.tar.bz2"
    "FC.tar.bz2"
    "M.tar.bz2"
    "MC.tar.bz2"
)

# Check if zip files are already present
missing_files=false
for file in "${FILES[@]}"; do
    if [ ! -f "${TARGET_DIR}/${file}" ]; then
        missing_files=true
        break
    fi
done

# If any zip file is not present, download them
if $missing_files; then
    for file in "${FILES[@]}"; do
        wget "${BASE_URL}${file}" -P "${TARGET_DIR}"
    done
fi

# Extract files
tar -xvf "${TARGET_DIR}/F.tar.bz2" -C "${TARGET_DIR}/dysarthria"
tar -xvf "${TARGET_DIR}/FC.tar.bz2" -C "${TARGET_DIR}/non_dysarthria"
tar -xvf "${TARGET_DIR}/M.tar.bz2" -C "${TARGET_DIR}/dysarthria"
tar -xvf "${TARGET_DIR}/MC.tar.bz2" -C "${TARGET_DIR}/non_dysarthria"
