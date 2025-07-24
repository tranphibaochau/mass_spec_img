import os
from PIL import Image
import numpy as np
import sys


def process_images(input_folder, output_folder, window_size=224, horizontal_stride=32, vertical_stride=192, coverage_threshold=0.7, min_x_distance=192):
    """
    Process all images in the input folder and save results to output folder. For a subfolder with large images, 
    the function will use a sliding window approach to extract regions of interest based on a computed common mask.

    Args:
        input_folder (str): Path to folder containing source images
        output_folder (str): Path to folder where processed images will be saved
        window_size (int): Size of the sliding window
        horizontal_stride (int): Horizontal stride of the sliding window
        vertical_stride (int): Vertical stride of the sliding window
        coverage_threshold (float): Threshold for the coverage of the sliding window
        min_x_distance (int): Minimum x distance between windows
    """
    # helper function to pad the image so dimensions are divisible by window_size
    def pad_image(img_array, window_size):
        h, w = img_array.shape
        pad_h = (window_size - (h % window_size)) % window_size
        pad_w = (window_size - (w % window_size)) % window_size
        return np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    # create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # get all files in the input directory
    subfolders = os.listdir(input_folder)
    for sf in subfolders:
        input_subfolder = os.path.join(input_folder, sf)
        
        # get all images in this subfolder
        image_files = [f for f in os.listdir(input_subfolder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # find mask for all images in this subfolder
        if len(image_files) > 0:
            # load all images to find the common mask
            images = []
            for f in image_files:
                img_path = os.path.join(input_subfolder, f)
                img = Image.open(img_path)
                images.append(np.array(img.convert('L')))
            

            # stack all images so we can find the common mask
            stacked_images = np.stack(images)
            # create mask where any image has non-zero pixels
            common_mask = np.any(stacked_images > 0, axis=0)
            # pad the mask so dimensions are divisible by window_size
            common_mask = pad_image(common_mask, window_size)
            valid_windows = [] # list of valid windows in case the original image is larger than 224x224
            if common_mask.shape[0] > 224 or common_mask.shape[1] > 224:    
                                
                img_width, img_height = common_mask.shape
                
                # sliding window with specified stride
                for y in range(0, img_height - window_size + 1, vertical_stride):
                    for x in range(0, img_width - window_size + 1, horizontal_stride):
                        # extract window
                        window = common_mask[y:y+window_size, x:x+window_size]
                        
                        # calculate coverage (percentage of non-black pixels)
                        non_black_pixels = np.sum(window > 0)
                        total_pixels = window_size * window_size
                        coverage = non_black_pixels / total_pixels
                        
                        # check if coverage meets threshold
                        if coverage > coverage_threshold:
                            # check if this window is far enough from previously selected windows
                            is_valid = True
                            # check if the window is far enough from the previously selected windows
                            if len(valid_windows) > 0 and abs(x - valid_windows[-1][0]) < min_x_distance and y == valid_windows[-1][1]:
                                is_valid = False
                            if is_valid:
                                valid_windows.append((x, y))
                                print(f"Valid window at ({x}, {y}) with coverage: {coverage:.3f} and distance: {abs(x - valid_windows[-1][0])}")
        
        # convert all images to black and white and resize it or cropt it into parts if the images are larger than 224x224
        for f in image_files:
            # Load image
            img_path = os.path.join(input_subfolder, f)
            img = Image.open(img_path)

            # convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # get the red channel only
            r, g, b = img.split()
            
            # use the red channel as the basis for the black and white image
            bw_img = r                    
            # get dimensions
            width, height = bw_img.size
            # if images are larger than 224x224 and have valid windows, crop the image into parts
            if len(valid_windows) > 1:
                bw_img_array = np.array(bw_img)
                padded_bw_img = pad_image(bw_img_array, window_size)
                bw_img = Image.fromarray(padded_bw_img)
                for i, window in enumerate(valid_windows):
                    output_subfolder = sf + "_" + str(i+1)
                    output_path = os.path.join(output_folder, output_subfolder)
                    os.makedirs(output_path, exist_ok=True)
                    x, y = window
                    output_img = bw_img.crop((x, y, x + window_size, y + window_size))
                    # save the cropped image into a separate part
                    filename = ".".join(f.split(".")[:-1]) + "_" + str(i+1) + ".png"
                    output_img.save(os.path.join(output_path, filename))
            elif len(valid_windows) == 1:
                # there is only one valid window, so we do not need to cut it into parts
                bw_img_array = np.array(bw_img)
                padded_bw_img = pad_image(bw_img_array, window_size)
                bw_img = Image.fromarray(padded_bw_img)
                output_path = os.path.join(output_folder, sf)
                os.makedirs(output_path, exist_ok=True)
                
                x, y = valid_windows[0]
                output_img = bw_img.crop((x, y, x + window_size, y + window_size))
                filename = ".".join(f.split(".")[:-1]) + ".png"
                output_img.save(os.path.join(output_path, filename))
            else:
                # create a square canvas
                output_img = Image.new('L', (window_size, window_size), 0)  # 'L' mode for grayscale, 0 for black background

                # calculate position to paste original (centered)
                paste_x = (window_size - width) // 2
                paste_y = (window_size - height) // 2

                # paste grayscale image onto square canvas
                output_img.paste(bw_img, (paste_x, paste_y))
                output_path = os.path.join(output_folder, sf)
                os.makedirs(output_path, exist_ok=True)
                filename = ".".join(f.split(".")[:-1]) + ".png"
                output_img.save(os.path.join(output_path, filename))

            print(f"Processed: {filename}")


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 224
    horizontal_stride = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    vertical_stride = int(sys.argv[5]) if len(sys.argv) > 5 else 192
    coverage_threshold = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    min_x_distance = int(sys.argv[7]) if len(sys.argv) > 7 else 192


    process_images(input_folder, output_folder, window_size, horizontal_stride, vertical_stride, coverage_threshold, min_x_distance)