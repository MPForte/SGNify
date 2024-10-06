#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nAfter registering at https://sgnify.is.tue.mpg.de/, provide your credentials:"
read -p "Username:" username
read -s -p "Password: " password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading SGNify..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=sgnify&resume=1&sfile=data.zip' -O 'data.zip' --no-check-certificate --continue
unzip data.zip -d data/
rm data.zip

# Create a temporary Git configuration to skip LFS
cat > ~/.gitconfig_nolfs << EOF
[filter "lfs"]
    smudge = git-lfs smudge --skip -- %f
    process = git-lfs filter-process --skip
[lfs]
    fetchexclude = *
EOF

# Function to run Git commands without LFS
git_nolfs() {
    GIT_CONFIG_GLOBAL=~/.gitconfig_nolfs GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.process= "$@"
}

# Update submodules without LFS
git_nolfs submodule update --init --recursive

# Create Conda environment
conda env create -f environment.yml 
eval "$(conda shell.bash hook)"

cd spectre
echo -e "\nDownload pretrained SPECTRE model..."
gdown --id 1vmWX6QmXGPnXTXWFgj67oHzOoOmxBh6B
mkdir -p pretrained/
mv spectre_model.tar pretrained/
cd ..

cp -r data/FLAME2020 spectre/data/FLAME2020

# Clean up temporary Git configuration
rm ~/.gitconfig_nolfs

echo -e "\nInstallation completed successfully!"
