import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import json
import random
import re

#Paths to files
CONFIG_FILEPATH = "config.json"
IMG_FILEPATH = "court.jpg"

#Color constants, do not edit
PAINT_IN = [0, 255, 0] #green
PAINT_OUT = [0, 0, 255] #blue
PAINT_LINES = [0, 0, 0] #black
FREETHROW_IN = [255, 0, 0] #red
FREETHROW_OUT = [0, 255, 255] #teal
THREEPT = [255, 0, 255] #magenta
CENTER = [255, 255, 0] #yellow
OUT_LINES = [128, 128, 128] #grey
FLOOR = [255, 255, 255] #white

BASE = np.array([PAINT_IN, PAINT_OUT, PAINT_LINES, FREETHROW_IN, FREETHROW_OUT, THREEPT, CENTER, OUT_LINES, FLOOR])

def get_desktop_path():
    # Get the desktop path for different OSes
    if os.name == 'nt':  # For Windows
        desktop_path = os.path.join(os.getenv('USERPROFILE'), 'Desktop')
    else:  # For macOS and Linux
        desktop_path = os.path.join(os.getenv('HOME'), 'Desktop')   
    return desktop_path

def replace_colors(img, new_colors):
    pixels = img.reshape(-1, 3)

    # Compute distances between each pixel and each base color
    base_colors_expanded = BASE[np.newaxis, :, :]
    pixels_expanded = pixels[:, np.newaxis, :]
    distances = np.sqrt(np.sum((pixels_expanded - base_colors_expanded) ** 2, axis=2))
    # Find the index of the closest base color for each pixel
    closest_base_color_indices = np.argmin(distances, axis=1)
    # Map replacement colors to each pixel based on closest base color
    new_pixels = new_colors[closest_base_color_indices]
    # Reshape new_pixels back to the original image shape
    new_img = new_pixels.reshape(img.shape)
    # Ensure new_img is of type uint8
    new_img = new_img.astype(np.uint8)
    return new_img

def resize_image(image, tile_size):
    tile_height, tile_width = tile_size
    img_height, img_width, _ = image.shape
    # Compute new dimensions to be divisible by tile size
    new_height = (img_height // tile_height) * tile_height
    new_width = (img_width // tile_width) * tile_width
    # Resize image
    if (new_height, new_width) != (img_height, img_width):
        resized_image = np.array(Image.fromarray(image).resize((new_width, new_height)))
    else:
        resized_image = image   
    return resized_image

def add_tiling(img, tile_size, thickness, shift=False):
    BLACK = [0, 0, 0]
    tile_height, tile_width = tile_size
    img_height, img_width, _ = img.shape
    mask = np.ones(img.shape)
    #add lengthwise tile lines
    for line in list(range(0, img_height, tile_height))+list(range(1, img_height, tile_height)):
        mask[line, :] = BLACK
    for line in list(range(0, img_width, tile_width))+list(range(1, img_width, tile_width)):
        mask[:, line] = BLACK
    if shift:
        mask = shift_tiles(mask, tile_size)
    img[np.where(np.all(mask == BLACK, axis=-1))] //= 6
    img[np.where(np.all(mask == BLACK, axis=-1))] *= 5

def split_into_tiles(image, tile_size):
    tile_height, tile_width = tile_size
    img_height, img_width, _ = image.shape
    # Calculate number of tiles in each dimension
    num_tiles_y = img_height // tile_height
    num_tiles_x = img_width // tile_width
    # Reshape the image into a 4D array of tiles
    tiles = (image.reshape(num_tiles_y, tile_height, num_tiles_x, tile_width, 3)
                  .swapaxes(1, 2)
                  .reshape(-1, tile_height, tile_width, 3))
    return tiles, num_tiles_y, num_tiles_x

def assemble_tiles(tiles, num_tiles_y, num_tiles_x, tile_size):
    tile_height, tile_width = tile_size
    img_height = num_tiles_y * tile_height
    img_width = num_tiles_x * tile_width
    # Reshape tiles back into the 4D array and merge into the final image
    new_image = (tiles.reshape(num_tiles_y, num_tiles_x, tile_height, tile_width, 3)
                        .swapaxes(1, 2)
                        .reshape(img_height, img_width, 3))
    return new_image

def shift_tiles(img, tile_size):
    tile_height, tile_width = tile_size
    img_height, img_width, _ = img.shape
    # Pad the image with half tile width on the right
    pad_width = tile_width // 2
    padded_img = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode='wrap')
    # Create an output image of the same size as the original
    shifted_img = np.zeros_like(img)
    # Apply shift for every second row
    for row in range(0, img_height, tile_height):
        if (row // tile_height) % 2 == 1:
            # Shift row by half the tile width
            shifted_img[row:row+tile_height] = padded_img[row:row+tile_height, pad_width:img_width+pad_width]
        else:
            # No shift for the other rows
            shifted_img[row:row+tile_height] = img[row:row+tile_height]
    return shifted_img
    
def gen_name(name: str) -> str: #Generate a number based on first and second letter of every token
    # Define a regular expression pattern to split on spaces, hyphens, and underscores
    pattern = r'[ \-_]+'
    tokens = re.split(pattern, name)
    ascii_values = [str(ord(token[0]))+token[1] for token in tokens if token]
    result = "".join(ascii_values)
    return result

def folder_name(name: str) -> str: #ensures valid folder names
    invalid_char_pattern = r'[<>:"/\\|?*\0-\31]'
    cleaned_name = re.sub(invalid_char_pattern, '_', name) 
    cleaned_name = cleaned_name.strip() 
    if not cleaned_name:
        cleaned_name = "default_folder" 
    return cleaned_name

def mkdir(path:str):
    if not os.path.exists(path):
    # Create the directory
        os.mkdir(path)

def slice():
    #Load config
    with open(CONFIG_FILEPATH, "r") as f:
        config = json.load(f)
    planks_width = config["planks_width"]
    planks_length = config["planks_length"]
    line_thickness = config["line_thickness"]
    seed = config["random_seed"]
    hardwood_color = config["hardwood_color"]
    if seed is None:
        seed = random.randInt()
    rng = np.random.default_rng(seed)
    courts_list = config["courts"]
    #Load image
    img = plt.imread(IMG_FILEPATH)
    tile_size = (img.shape[0]//planks_length, img.shape[1]//planks_width)
    # Ensure img is writable
    img = np.copy(img)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    # Resize the image if necessary
    img = resize_image(img, tile_size)
    #get base directory
    basedir = os.path.join(get_desktop_path(), config["folder_name"])
    mkdir(basedir)

    for court in courts_list:
        #make directory
        court_dir = os.path.join(basedir, folder_name(court["name"]))
        mkdir(court_dir)
        study_name = gen_name(court["name"])
        for use_hardwood in range(2):
            if use_hardwood:
                court["floor"] = hardwood_color
            new_colors = np.array([value for key, value in court.items() if key!="name"])
            #Format image
            rep_img = replace_colors(img, new_colors)
            base_img = np.copy(rep_img)
            add_tiling(base_img, tile_size, line_thickness, shift=True)
            add_tiling(rep_img, tile_size, line_thickness, shift=False)
            # Split the image into tiles
            tiles, num_tiles_y, num_tiles_x = split_into_tiles(rep_img, tile_size)
            tiles_copy = np.copy(tiles)

            # Shuffle tiles
            rng.shuffle(tiles)
            np.random.shuffle(tiles_copy)

            # Assemble the shuffled tiles back into a single image
            shuffled_seed = assemble_tiles(tiles, num_tiles_y, num_tiles_x, tile_size)
            shuffled_random = assemble_tiles(tiles_copy, num_tiles_y, num_tiles_x, tile_size)
            shuffled_seed = shift_tiles(shuffled_seed, tile_size)
            shuffled_random = shift_tiles(shuffled_random, tile_size)

            # Convert to PIL image
            og_img = Image.fromarray(base_img)
            img_seed = Image.fromarray(shuffled_seed)
            img_random = Image.fromarray(shuffled_random)
            
            # Save or display the image
            suffix = "_hw" if use_hardwood else ""
            og_img.save(os.path.join(court_dir, f"{study_name}_court{suffix}.png"))
            img_seed.save(os.path.join(court_dir, f"{study_name}_seed{seed}{suffix}.png"))
            img_random.save(os.path.join(court_dir, f"{study_name}_random{suffix}.png"))

if __name__ == "__main__":
    slice()