import os
import cv2
import numpy as np

from boxstat import yolov8_annotations_folder


def split_image_and_adjust_labels(image_path, label_path, output_dir, tile_size, overlap_fraction):
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}. Please check the file path and format.")

    H, W, _ = image.shape
    w, h = tile_size
    overlap_w = int(w * overlap_fraction)
    overlap_h = int(h * overlap_fraction)

    # Read YOLO annotations
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                annotations.append((cls, x_center, y_center, width, height))

    os.makedirs(output_dir, exist_ok=True)

    tile_id = 0
    for y in range(0, H, h - overlap_h):
        for x in range(0, W, w - overlap_w):
            tile = image[y:min(y + h, H), x:min(x + w, W)]
            tile_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.png"
            cv2.imwrite(os.path.join(output_dir, tile_name), tile)

            # Adjust annotations
            tile_labels = []
            for cls, xc, yc, bw, bh in annotations:
                abs_xc = xc * W
                abs_yc = yc * H
                abs_bw = bw * W
                abs_bh = bh * H

                if (x <= abs_xc <= x + w) and (y <= abs_yc <= y + h):
                    new_xc = (abs_xc - x) / w
                    new_yc = (abs_yc - y) / h
                    new_bw = abs_bw / w
                    new_bh = abs_bh / h

                    tile_labels.append(f"{int(cls)} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}")

            with open(
                    os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.txt"),
                    'w') as f:
                f.write('\n'.join(tile_labels))

            tile_id += 1


def process_folder(input_dir, output_dir, tile_size, overlap_fraction):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            label_path = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}.txt")
            split_image_and_adjust_labels(image_path, label_path, output_dir, tile_size, overlap_fraction)


def resize_to_fit(image, max_width, max_height):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)))

def visualize_annotations(image_base_name, input_dir="path/to/input_folder", output_dir="output_tiles"):
    image_path = os.path.join(input_dir, f"{image_base_name}.png")
    label_path = os.path.join(input_dir, f"{image_base_name}.txt")
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}. Please check the file path and format.")

    H, W, _ = image.shape

    # Draw annotations on the original image
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                x1 = int((x_center - width / 2) * W)
                y1 = int((y_center - height / 2) * H)
                x2 = int((x_center + width / 2) * W)
                y2 = int((y_center + height / 2) * H)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(int(cls)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display original image
    cv2.imshow(f"Original Annotations for {image_base_name}", image)

    # Create mosaic for tiled images
    tile_images = []
    tile_files = sorted([f for f in os.listdir(output_dir) if f.startswith(image_base_name) and f.endswith(".png")],
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Determine the grid size for mosaic
    grid_cols = int(np.ceil(np.sqrt(len(tile_files))))
    grid_rows = int(np.ceil(len(tile_files) / grid_cols))

    # Find the max dimensions for padding
    max_height = max(cv2.imread(os.path.join(output_dir, f)).shape[0] for f in tile_files)
    max_width = max(cv2.imread(os.path.join(output_dir, f)).shape[1] for f in tile_files)

    # Load, annotate, and pad the tiles
    for tile_file in tile_files:
        tile_image = cv2.imread(os.path.join(output_dir, tile_file))
        tile_label_file = os.path.join(output_dir, f"{os.path.splitext(tile_file)[0]}.txt")

        if os.path.exists(tile_label_file):
            with open(tile_label_file, 'r') as f:
                annotations = [line.strip().split() for line in f.readlines()]
                for ann in annotations:
                    cls, x_center, y_center, width, height = map(float, ann)
                    h, w, _ = tile_image.shape
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    cv2.rectangle(tile_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(tile_image, str(int(cls)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Pad tile to match max dimensions
        padded_tile = cv2.copyMakeBorder(
            tile_image,
            0, max_height - tile_image.shape[0],
            0, max_width - tile_image.shape[1],
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        tile_images.append(padded_tile)

    # Fill incomplete grid with black tiles if necessary
    while len(tile_images) < grid_rows * grid_cols:
        tile_images.append(np.zeros((max_height, max_width, 3), dtype=np.uint8))

    # Arrange tiles in a grid
    mosaic_rows = [np.hstack(tile_images[i*grid_cols:(i+1)*grid_cols]) for i in range(grid_rows)]
    mosaic = np.vstack(mosaic_rows)

    # Resize images to fit the screen side by side
    screen_width = 1920  # Example screen width, adjust as needed
    screen_height = 1080  # Example screen height, adjust as needed
    half_screen_width = screen_width // 2

    resized_image = resize_to_fit(image, half_screen_width, screen_height)
    resized_mosaic = resize_to_fit(mosaic, half_screen_width, screen_height)

    # Display images side by side
    cv2.imshow(f"Original Annotations for {image_base_name}", resized_image)
    cv2.imshow(f"Tiled Mosaic for {image_base_name}", resized_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    folder = "/home/joakim/data/challenge"
    yolo_folder = os.path.join(folder, "test_annotations")
    output_dir = os.path.join(folder, "test_annotations_tiled")
    tile_size = (640, 640)  # Tile size (w, h)
    overlap_fraction = 0.2  # 20% overlap

    process_folder(yolo_folder, output_dir, tile_size, overlap_fraction)

    # Visualization example
    visualize_annotations("2019_09_02_C0002_3700_mavic_fr709", input_dir=yolo_folder, output_dir=output_dir)
