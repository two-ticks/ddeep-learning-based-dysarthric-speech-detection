
# Define the URLs and target directory
urls = ("http://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2" "http://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2" "http://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2" "http://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2")

target_dir="/scratch/TORGO"

# Check if zip files are already present
all_present=true
for url in "${urls[@]}"; do
    if [ ! -f "${target_dir}/${url##*/}" ]; then
        all_present=false
        break
    fi
done

# If any zip file is not present, download them
if ! $all_present; then
    for url in "${urls[@]}"; do
        if [ ! -f "${target_dir}/${url##*/}" ]; then
            wget "$url" -P "$target_dir"
        fi
    done
fi

# Extract files
tar -xvf "${target_dir}/F.tar.bz2" -C "${target_dir}/dysarthria"
tar -xvf "${target_dir}/FC.tar.bz2" -C "${target_dir}/non_dysarthria"
tar -xvf "${target_dir}/M.tar.bz2" -C "${target_dir}/dysarthria"
tar -xvf "${target_dir}/MC.tar.bz2" -C "${target_dir}/non_dysarthria"
