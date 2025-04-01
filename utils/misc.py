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
    Restore original filenames and reorganize output files.
    
    Parameters:
    output_folder (Path): Path to the output folder
    result_folder (Path): Path to the temporary result folder
    """
    import json
    import shutil
    from pathlib import Path
    
    output_folder = Path(output_folder)
    result_folder = Path(result_folder)
    
    # Filename mapping path
    mapping_path = result_folder / "filename_map.json"
    
    # Check if mapping file exists
    if not mapping_path.exists():
        print(f"No filename mapping found at {mapping_path}")
        return
    
    # Read filename mapping
    with mapping_path.open('r') as f:
        filename_map = json.load(f)
    
    # Restore image filenames
    images_folder = output_folder / "images"
    if images_folder.exists():
        for seq_name, original_name in filename_map.items():
            seq_path = images_folder / seq_name
            if seq_path.exists():
                # Create path for original filename
                original_path = images_folder / original_name
                
                # Remove original file if it exists
                if original_path.exists():
                    original_path.unlink()
                
                # Rename sequential file to original name
                seq_path.rename(original_path)
        
        print(f"Restored original filenames in {images_folder}")
    
    # Function to restore filenames for a specific folder and extension
    def restore_folder_filenames(folder, extension):
        if folder.exists():
            # Create a mapping from sequential names to original names
            reverse_map = {f"{int(k.split('.')[0]):03d}.{extension}": v for k, v in filename_map.items()}
            
            files = sorted(folder.glob(f"*.{extension}"))
            for file in files:
                seq_name = file.name
                if seq_name in reverse_map:
                    original_name = reverse_map[seq_name]
                    original_path = folder / original_name.replace('.jpg', '.' + extension)
                    
                    # Remove original file if it exists
                    if original_path.exists():
                        original_path.unlink()
                    
                    # Rename sequential file to original name
                    file.rename(original_path)
            
            print(f"Restored original filenames in {folder}")
    
    # Restore pkl and obj filenames
    restore_folder_filenames(output_folder / "results", "pkl")
    restore_folder_filenames(output_folder / "meshes", "obj")
    
    # Remove images_sequential folder if it exists
    sequential_folder = output_folder / "images_sequential"
    if sequential_folder.exists():
        shutil.rmtree(sequential_folder)

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