import glob
import os
import pickle
import shutil
from subprocess import run
from pathlib import Path
import numpy as np
from PIL import Image
import re


def get_end(result_path):
    # Simply count the number of image files
    return len(list(result_path.joinpath("images").glob("*.png")))


def get_filename_by_index(folder_path, index):
    """
    Get the filename at a specific index in the sorted list of files.
    
    Parameters:
    folder_path (Path): Path to the image folder
    index (int): 1-based index of the file
    
    Returns:
    str: Filename at the given index
    """
    # Get a sorted list of all image files
    files = sorted([f.name for f in folder_path.glob("*.png")])
    
    # Return the filename at index-1 (convert from 1-based to 0-based)
    # or fall back to sequential naming if index is out of range
    if 0 < index <= len(files):
        return files[index-1]
    else:
        return f"{index:03}.png"


def extract_frames(*, video_path, output_folder):
    return run(
        ["ffmpeg", "-i", video_path, output_folder.joinpath("%03d.png"), "-nostdin"],
        check=True,
    )


def copy_frames(*, image_dir_path, output_folder):
    """
    Copy frames from a folder to another, extracting numbers from original filenames
    and using those exact numbers for the new filenames.
    
    Parameters:
    image_dir_path (Path): Path to the folder with the images
    output_folder (Path): Path to the folder where frames will be copied
    
    Raises:
    ValueError: If any filename doesn't contain a number
    """
    
    image_dir_path = Path(image_dir_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
        
    patterns = ('*.png', '*.jpg', '*.jpeg')
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.iglob(os.path.join(str(image_dir_path), pattern)))
    
    # Process each file
    for imgfile in sorted(image_files):
        original_name = os.path.basename(imgfile)
        
        # Extract numbers from the filename
        numbers = re.findall(r'\d+', original_name)
        if numbers:
            # Use the last group of numbers (often the frame number)
            number = numbers[-1]
            # Ensure it's a 3-digit format
            new_name = f"{int(number):03}.png"
            
            # If already PNG, just copy, otherwise convert it
            if imgfile.lower().endswith('.png'):
                shutil.copy(imgfile, output_folder.joinpath(new_name))
            else:
                img = Image.open(imgfile)
                img.save(output_folder.joinpath(new_name), "PNG")
        else:
            # Raise an error if no numbers are found
            raise ValueError(f"Error: No numbers found in filename: {original_name}. All filenames must contain numeric values.")
                
    return


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