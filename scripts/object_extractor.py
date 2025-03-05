import cv2
import numpy as np
from skimage import measure
import os
from PIL import Image
import sys


def find_bounding_boxes(mask_path, visualization_path=None):
    """
    Given a mask image, find the bounding boxes of objects inside it. Each object must be horizontally separated.

    Args:
        mask_path (str): Path to the binary mask image
        visualization_path (str, optional): Path to save visualization of detected objects

    Returns:
        list: List of bounding boxes as (x, y, width, height) tuples
    """
    # load binary mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask image from {mask_path}")

    # project mask horizontally (sum along vertical axis)
    horizontal_projection = np.sum(mask, axis=0)

    # find horizontal boundaries where projection changes from 0 to non-zero and vice versa
    boundaries = []
    in_object = False

    for x, col_sum in enumerate(horizontal_projection):
        if not in_object and col_sum > 0:
            # Start of an object
            start_x = x
            in_object = True
        elif in_object and col_sum == 0:
            # End of an object
            end_x = x
            boundaries.append((start_x, end_x))
            in_object = False

    # handle case where last object extends to the end of the image
    if in_object:
        boundaries.append((start_x, len(horizontal_projection)))

    # extract bounding boxes for each horizontal segment
    bounding_boxes = []

    # create visualization if requested
    if visualization_path:
        vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for i, (start_x, end_x) in enumerate(boundaries):
        # extract this horizontal segment
        segment = mask[:, start_x:end_x]

        # Find the vertical extents
        vertical_projection = np.sum(segment, axis=1)
        non_zero_rows = np.nonzero(vertical_projection)[0]

        if len(non_zero_rows) == 0:
            continue

        start_y = non_zero_rows[0]
        end_y = non_zero_rows[-1]

        # Store bbox as (x, y, width, height)
        bbox = (start_x, start_y, end_x - start_x, end_y - start_y + 1)
        bounding_boxes.append(bbox)

        # Add to visualization
        if visualization_path:
            x, y, w, h = bbox
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_img, f"#{i + 1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # save visualization if requested
    if visualization_path:
        cv2.imwrite(visualization_path, vis_img)

    return bounding_boxes




def get_object_coordinates(mask_path, output_visualization=None):
    """
    A wrapper function that extracts bounding boxes objects in a mask image.

    Args:
        mask_path (str): Path to the binary mask image
        output_visualization (str, optional): Path to save visualization

    Returns:
        list: List of dictionaries containing object coordinates and dimensions
    """
    # Get bounding boxes
    bboxes = find_bounding_boxes(
        mask_path,
        visualization_path=output_visualization
    )

    # convert to a more structured format
    object_coordinates = []

    # for each detected object, only store its top left and bottom right coordinates
    for i, (x, y, w, h) in enumerate(bboxes):
        object_coordinates.append(([x,y], [x + w, y + h]))
    return object_coordinates


def extract_objects(image_path, coordinates, output_size, output_path):
    """
        A function that extracts images of object given the coordinates of the bounding boxes

        Args:
            image_path (str): Path to the folder that contains the images
            coordinates (list): list of tuples that contains top-left and bottom-right coordinates of each bounding box
        Returns:
            None (saves images to disk)
    """

    # make sure output_path is created
    os.makedirs(output_path, exist_ok=True)

    images = [f for f in os.listdir(image_path) if f.endswith(".png")]  # find list of images in the folder
    for img in images:
        print("Cropping:", img)
        # Open the original image
        original_image = Image.open(os.path.join(image_path, img))

        for i, (top_left, bottom_right) in enumerate(coordinates):
            # ignore the first tissue
            if i > 0:
                max_output_size = output_size
                # crop the image using the top_left and bottom_right coordinates
                cropped_image = original_image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

                cropped_width, cropped_height = cropped_image.size
                # the bg image has to be larger than the cropped image before resizing
                max_output_size = max(max_output_size, max(cropped_width, cropped_height))
                # calculate the position to center the cropped image on the black background
                paste_position = ((max_output_size - cropped_width) // 2, (max_output_size - cropped_height) // 2)

                new_image = Image.new('RGB', (max_output_size, max_output_size), (0, 0, 0))  # Black background

                # paste the cropped image to the black image
                new_image.paste(cropped_image, paste_position)
                # resize it to the desired size, if needed
                if max_output_size > output_size:
                    new_image = new_image.resize((output_size, output_size))
                # convert to grey scale image before saving
                new_image.convert("L").save(os.path.join(output_path, f"{img}_s{i}_{output_size}x{output_size}.png"))
    return



"""mask_path = "/home/mass_spec_img/mass_spec_img/data/raw_img_v2/masks/SQ1636_mask/SQ1636_mask.png"
image_path = "/home/mass_spec_img/mass_spec_img/data/raw_img_v2/SQ1636_images"
output = "/home/mass_spec_img/mass_spec_img/data/cropped/SQ1636_v2_cropped""""

mask_path = sys.argv[1]
image_path = sys.argv[2]
output_path = sys.argv[3]
output_size = sys.argv[4]

# get object coordinates
object_coords = get_object_coordinates(mask_path)
file_name = mask_path.split("/")[-1]
# print results
print(f"Detected {len(object_coords)} objects in {file_name}:")
for i, (top_left, bottom_right) in enumerate(object_coords):
    print(f"Object #{i+1}:")
    print(f"  Top Left: ({top_left[0]}, {top_left[1]})")
    print(f"  Bottom-right: ({bottom_right[0]}, {bottom_right[1]})")

extract_objects(image_path, object_coords, output_size, output_path)
