import cv2
import numpy as np
import os
import json
import argparse
from PIL import Image


def get_horizontal_object_corners(mask_path, min_size=100, visualization_path=None):
    """
    Extract the four corners of each horizontally separated object from a binary mask.

    Args:
        mask_path (str): Path to the binary mask image
        min_size (int): Minimum object size in pixels
        visualization_path (str, optional): Path to save visualization

    Returns:
        list: List of dictionaries containing object data with corner coordinates
    """
    # Load binary mask image
    if isinstance(mask_path, str):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask image from {mask_path}")
    else:
        # Assume mask_path is already a numpy array
        mask = mask_path.copy()

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

    # create visualization if requested
    if visualization_path:
        vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    objects_data = []

    for i, (start_x, end_x) in enumerate(boundaries):
        # extract this horizontal segment
        segment = mask[:, start_x:end_x]

        # skip if segment is too small
        if np.sum(segment) / 255 < min_size:
            continue

        # find the vertical extents
        vertical_projection = np.sum(segment, axis=1)
        non_zero_rows = np.nonzero(vertical_projection)[0]

        if len(non_zero_rows) == 0:
            continue

        start_y = non_zero_rows[0]
        end_y = non_zero_rows[-1]

        # calculate the four corners
        top_left = (start_x, start_y)
        top_right = (end_x - 1, start_y)  # Subtract 1 for proper indexing
        bottom_right = (end_x - 1, end_y)
        bottom_left = (start_x, end_y)

        # store object data
        obj_data = {
            'id': i + 1,
            'corners': {
                'top_left': top_left,
                'top_right': top_right,
                'bottom_right': bottom_right,
                'bottom_left': bottom_left
            },
            # Also include traditional bbox format for convenience
            'bbox': {
                'x': start_x,
                'y': start_y,
                'width': end_x - start_x,
                'height': end_y - start_y + 1,
                'x2': end_x,
                'y2': end_y + 1
            }
        }
        objects_data.append(obj_data)

        # add to visualization
        if visualization_path:
            # draw bounding box
            cv2.rectangle(vis_img, top_left, (end_x, end_y + 1), (0, 255, 0), 2)

            # draw and label corners
            cv2.circle(vis_img, top_left, 3, (255, 0, 0), -1)
            cv2.circle(vis_img, top_right, 3, (0, 0, 255), -1)
            cv2.circle(vis_img, bottom_right, 3, (255, 0, 255), -1)
            cv2.circle(vis_img, bottom_left, 3, (0, 255, 255), -1)

            # add object ID
            cv2.putText(vis_img, f"#{i + 1}", (start_x, start_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # save visualization if requested
    if visualization_path:
        cv2.imwrite(visualization_path, vis_img)

    return objects_data


def extract_objects_with_corners(mask_path, original_path=None, output_dir=None, min_size=100):
    """
    Extract horizontally separated objects and their corner coordinates from a mask.
    Optionally extract the corresponding regions from an original image.

    Args:
        mask_path (str): Path to the binary mask image
        original_path (str, optional): Path to the original image
        output_dir (str, optional): Directory to save extracted objects and data
        min_size (int): Minimum object size in pixels

    Returns:
        list: List of objects with corner coordinates
    """
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set visualization path if output directory is provided
    vis_path = None
    if output_dir:
        vis_path = os.path.join(output_dir, "detection_mask.png")

    # get object corners
    objects_data = get_horizontal_object_corners(mask_path, min_size, vis_path)

    # load original image if provided
    if original_path and output_dir:
        # try with OpenCV first
        original_cv = cv2.imread(original_path)

        if original_cv is None:
            print(f"Warning: Could not load original image with OpenCV. Trying PIL...")
            try:
                # try with PIL
                original_pil = Image.open(original_path)
                # convert PIL to OpenCV format
                original_cv = np.array(original_pil)
                original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error loading original image with PIL: {e}")
                original_cv = None

        if original_cv is not None:
            # create a copy for visualization
            original_vis = original_cv.copy()

            # extract objects from original image
            for obj in objects_data:
                # get bounding box
                bbox = obj['bbox']
                x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']

                # sxtract object using bounding box
                object_crop = original_cv[y:y + height, x:x + width]

                # save extracted object
                cv2.imwrite(os.path.join(output_dir, f"object_{obj['id']}_original.png"), object_crop)

                # draw on visualization
                corners = obj['corners']

                # draw bounding box
                cv2.rectangle(original_vis, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # draw corners
                cv2.circle(original_vis, corners['top_left'], 3, (255, 0, 0), -1)
                cv2.circle(original_vis, corners['top_right'], 3, (0, 0, 255), -1)
                cv2.circle(original_vis, corners['bottom_right'], 3, (255, 0, 255), -1)
                cv2.circle(original_vis, corners['bottom_left'], 3, (0, 255, 255), -1)

                # add object ID
                cv2.putText(original_vis, f"#{obj['id']}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # save visualization
            cv2.imwrite(os.path.join(output_dir, "objects_detected.png"), original_vis)

            # create a side-by-side comparison
            if os.path.exists(vis_path):
                mask_vis = cv2.imread(vis_path)
                h, w = mask_vis.shape[:2]

                # resize original visualization to match mask visualization
                if original_vis.shape[:2] != (h, w):
                    original_vis = cv2.resize(original_vis, (w, h))

                comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison[:, :w] = mask_vis
                comparison[:, w:] = original_vis
                cv2.imwrite(os.path.join(output_dir, "comparison.png"), comparison)

    # save object data as JSON if output directory is provided
    if output_dir:
        # convert data to serializable format (tuples to lists)
        serializable_data = []
        for obj in objects_data:
            serializable_obj = {
                'id': obj['id'],
                'corners': {
                    'top_left': list(obj['corners']['top_left']),
                    'top_right': list(obj['corners']['top_right']),
                    'bottom_right': list(obj['corners']['bottom_right']),
                    'bottom_left': list(obj['corners']['bottom_left'])
                },
                'bbox': obj['bbox']
            }
            serializable_data.append(serializable_obj)

        # save as JSON
        with open(os.path.join(output_dir, "objects_data.json"), "w") as f:
            json.dump(serializable_data, f, indent=2)

        # also save as CSV for easier import into other tools
        with open(os.path.join(output_dir, "objects_data.csv"), "w") as f:
            f.write(
                "id,x,y,width,height,x2,y2,top_left_x,top_left_y,top_right_x,top_right_y,bottom_right_x,bottom_right_y,bottom_left_x,bottom_left_y\n")
            for obj in objects_data:
                corners = obj['corners']
                bbox = obj['bbox']
                f.write(
                    f"{obj['id']},{bbox['x']},{bbox['y']},{bbox['width']},{bbox['height']},{bbox['x2']},{bbox['y2']},{corners['top_left'][0]},{corners['top_left'][1]},{corners['top_right'][0]},{corners['top_right'][1]},{corners['bottom_right'][0]},{corners['bottom_right'][1]},{corners['bottom_left'][0]},{corners['bottom_left'][1]}\n")

    return objects_data


def process_folder(masks_folder, originals_folder=None, output_base_dir="output", min_size=100):
    """
    Process all mask images in a folder and extract corner coordinates.

    Args:
        masks_folder (str): Path to folder containing mask images
        originals_folder (str, optional): Path to folder containing original images
        output_base_dir (str): Base directory to save results
        min_size (int): Minimum object size in pixels
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Get all mask files
    mask_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        mask_files.extend([f for f in os.listdir(masks_folder) if f.lower().endswith(ext)])

    # Process each mask file
    for mask_file in mask_files:
        mask_path = os.path.join(masks_folder, mask_file)
        base_name = os.path.splitext(mask_file)[0]

        # Create output directory for this image
        image_output_dir = os.path.join(output_base_dir, base_name)

        # Find corresponding original image if folder is provided
        original_path = None
        if originals_folder:
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                potential_path = os.path.join(originals_folder, base_name + ext)
                if os.path.exists(potential_path):
                    original_path = potential_path
                    break

        print(f"Processing: {mask_path}")
        if original_path:
            print(f"  with original: {original_path}")

        # Extract objects
        try:
            objects = extract_objects_with_corners(
                mask_path,
                original_path,
                image_output_dir,
                min_size
            )

            # Print results
            print(f"  Detected {len(objects)} objects")
            for obj in objects:
                corners = obj['corners']
                print(f"    Object #{obj['id']}:")
                print(f"      Top-Left: {corners['top_left']}")
                print(f"      Top-Right: {corners['top_right']}")
                print(f"      Bottom-Right: {corners['bottom_right']}")
                print(f"      Bottom-Left: {corners['bottom_left']}")

        except Exception as e:
            print(f"  Error processing {mask_file}: {e}")

    print(f"Processing complete. Results saved to {output_base_dir}")


def process_pil_image(pil_image, min_size=100):
    """
    Process a PIL image directly to get object corners.

    Args:
        pil_image (PIL.Image): PIL Image object of the mask
        min_size (int): Minimum object size in pixels

    Returns:
        list: List of objects with corner coordinates
    """
    # Convert PIL image to numpy array
    mask_np = np.array(pil_image)

    # If it's an RGB image, convert to grayscale
    if len(mask_np.shape) == 3:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

    # Get object corners
    return get_horizontal_object_corners(mask_np, min_size)


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract object corners from binary masks')
    parser.add_argument('--masks', required=True, help='Path to folder containing mask images or a single mask image')
    parser.add_argument('--originals', help='Path to folder containing original images or a single original image')
    parser.add_argument('--output', default='output', help='Directory to save extracted objects')
    parser.add_argument('--min-size', type=int, default=100, help='Minimum object size in pixels')

    args = parser.parse_args()

    # Check if the masks argument is a folder or a single file
    if os.path.isdir(args.masks):
        # Process entire folder
        process_folder(args.masks, args.originals, args.output, args.min_size)
    else:
        # Process single file
        original_path = args.originals if args.originals and os.path.isfile(args.originals) else None
        output_dir = args.output

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        objects = extract_objects_with_corners(
            args.masks,
            original_path,
            output_dir,
            args.min_size
        )

        print(f"Detected {len(objects)} objects")
        for obj in objects:
            corners = obj['corners']
            print(f"Object #{obj['id']}:")
            print(f"  Top-Left: {corners['top_left']}")
            print(f"  Top-Right: {corners['top_right']}")
            print(f"  Bottom-Right: {corners['bottom_right']}")
            print(f"  Bottom-Left: {corners['bottom_left']}")