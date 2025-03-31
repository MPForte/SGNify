import glob
import os
import pickle
import shutil
from subprocess import run
from pathlib import Path
import numpy as np
from PIL import Image
import re
import json
from collections import OrderedDict


def get_end(result_path):
    # Simply count the number of image files
    return len(list(result_path.joinpath("images").glob("*.png")))

def extract_frames(*, video_path, output_folder):
    return run(
        ["ffmpeg", "-i", video_path, output_folder.joinpath("%03d.png"), "-nostdin"],
        check=True,
    )

def copy_frames(*, image_dir_path, output_folder):    
    # Create mapping dictionary
    filename_map = OrderedDict()
    
    i = 1
    patterns = ('*.png', '*.jpg', '*.jpeg')
    
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.iglob(os.path.join(str(image_dir_path), pattern)))
    
    for imgfile in sorted(image_files):
        seq_name = f"{i:03}.png"
        original_name = os.path.basename(imgfile)
        
        # Save mapping between sequential and original names
        filename_map[seq_name] = original_name
        
        # Use sequential numbering for processing
        output_path = output_folder.joinpath(seq_name)
        
        # If already PNG, just copy, otherwise convert it
        if imgfile.lower().endswith('.png'):
            shutil.copy(imgfile, output_path)
        else:
            img = Image.open(imgfile)
            img.save(output_path, "PNG")
        
        i += 1
    
    # Save the mapping to a json file
    mapping_path = output_folder.parent / "filename_map.json"
    with mapping_path.open('w') as f:
        json.dump(filename_map, f, indent=2)

    return

def restore_original_filenames(*, output_folder, result_folder):
    """
    Restore original filenames in the output folder.
    
    Parameters:
    output_folder (Path): Path to the output folder
    result_folder (Path): Path to the temporary result folder
    """
    import json
    import shutil
    
    output_folder = Path(output_folder)
    result_folder = Path(result_folder)
    
    mapping_path = result_folder / "filename_map.json"
    if not mapping_path.exists():
        print("No filename mapping found, keeping sequential filenames")
        return
    
    with mapping_path.open('r') as f:
        filename_map = json.load(f)
    
    output_images_folder = output_folder / "images"
    if not output_images_folder.exists():
        print("No output images folder found")
        return
    
    # Create a backup folder
    backup_folder = output_folder / "images_sequential"
    backup_folder.mkdir(exist_ok=True)
    
    # Copy files with original names
    for seq_name, original_name in filename_map.items():
        seq_path = output_images_folder / seq_name
        if seq_path.exists():
            # Make a backup
            shutil.copy(seq_path, backup_folder / seq_name)
            
            # Create new file with original name
            original_path = output_images_folder / original_name
            if original_path.exists():
                original_path.unlink()
            shutil.copy(seq_path, original_path)
            seq_path.unlink()  # Remove sequential file
    
    print(f"Restored original filenames in {output_images_folder}")
    print(f"Backup of sequential files saved in {backup_folder}")

def create_video(*, images_folder, output_path):
    return run(
        ["ffmpeg", "-r", "30", "-pattern_type", "glob", "-i", images_folder, "-y", output_path, "-nostdin"], check=True
    )


def compute_betas(*, rps_folder, beta_path):
    betas = []
    for hand in ["left", "right"]:
        pkl_files = list(rps_folder.joinpath(hand, "results").glob("*.pkl"))
        for pkl_file in pkl_files:
            with pkl_file.open("rb") as file:
                betas.append(pickle.load(file)["betas"][0])

    with beta_path.open("wb") as file:
        pickle.dump(np.median(betas, axis=0), file, pickle.HIGHEST_PROTOCOL)