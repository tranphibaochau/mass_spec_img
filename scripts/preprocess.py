import os
from PIL import Image
import numpy as np
import sys

def process_images(input_folder, output_folder):
    """
    Process all images in the input folder and save results to output folder.

    Args:
        input_folder (str): Path to folder containing source images
        output_folder (str): Path to folder where processed images will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all files in the input directory
    subfolders = os.listdir(input_folder)
    print("Subfolders:", subfolders)

    for sf in subfolders:
            subfolder_output = os.path.join(output_folder, sf)
            print("creating subfolder output:", subfolder_output)
            os.makedirs(subfolder_output, exist_ok=True)
            for filename in os.listdir(os.path.join(input_folder,sf)):
                # Check if file is an image (simple check, extend if needed)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Load image
                    img_path = os.path.join(input_folder, sf, filename)
                    img = Image.open(img_path)

                    # Convert to RGB if not already
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Get the red channel only
                    r, g, b = img.split()
                    
                    # Use the red channel as the basis for the black and white image
                    # No need to change anything since we want highest red to be white
                    bw_img = r
                    
                    # Get dimensions
                    width, height = bw_img.size
                    max_dim = max(width, height)

                    # Create a square canvas with the max dimension
                    square_img = Image.new('L', (max_dim, max_dim), 0)  # 'L' mode for grayscale, 0 for black background

                    # Calculate position to paste original (centered)
                    paste_x = (max_dim - width) // 2
                    paste_y = (max_dim - height) // 2

                    # Paste grayscale image onto square canvas
                    square_img.paste(bw_img, (paste_x, paste_y))

                    # Resize to 224x224
                    final_img = square_img.resize((224, 224), Image.LANCZOS)  # LANCZOS for high-quality downsampling

                    # Save the processed image
                    output_path = os.path.join(subfolder_output, filename)
                    final_img.save(output_path)

                    print(f"Processed: {filename}")


# Example usage
if __name__ == "__main__":
    input_folder = sys.argv[1]  # Change this to your input folder
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

    process_images(input_folder, output_folder)