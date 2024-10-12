#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nAfter registering at https://sgnify.is.tue.mpg.de/, provide your credentials:"
read -p "Username:" username
read -s -p "Password: " password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading SGNify..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=sgnify&resume=1&sfile=data.zip' -O 'data.zip' --continue

conda env create -f environment.yml 
eval "$(conda shell.bash hook)"
conda activate sgnify

unzip data.zip
rm data.zip

git submodule update --init --recursive --force
git submodule foreach --recursive git lfs install
git lfs pull -I "./spectre/external/face_detection/ibug/face_detection/retina_face/weights"

pip install --no-deps pyrender==0.1.23
pip install -e ./spectre/external/face_alignment
pip install -e ./spectre/external/face_detection

cd spectre
echo -e "\nDownload pretrained SPECTRE model..."
gdown --id 1vmWX6QmXGPnXTXWFgj67oHzOoOmxBh6B
mkdir -p pretrained/
mv spectre_model.tar pretrained/
cd ..

cp -r data/FLAME2020 spectre/data/FLAME2020
