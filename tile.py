import os
import cv2
import numpy as np

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
            # Adjust tile position to ensure it fits within the original image
            if y + h > H:
                y = H - h
            if x + w > W:
                x = W - w

            # Adjust annotations
            tile_labels = []
            for cls, xc, yc, bw, bh in annotations:
                abs_xc = xc * W
                abs_yc = yc * H
                abs_bw = bw * W
                abs_bh = bh * H

                # Bounding box coordinates
                x1 = abs_xc - abs_bw / 2
                y1 = abs_yc - abs_bh / 2
                x2 = abs_xc + abs_bw / 2
                y2 = abs_yc + abs_bh / 2

                # Check for overlap with the tile
                if (x1 < x + w and x2 > x and y1 < y + h and y2 > y):
                    # Trim bounding box to fit within the tile
                    x1_clipped = max(x1, x)
                    y1_clipped = max(y1, y)
                    x2_clipped = min(x2, x + w)
                    y2_clipped = min(y2, y + h)

                    # Convert back to YOLO format
                    new_xc = ((x1_clipped + x2_clipped) / 2 - x) / w
                    new_yc = ((y1_clipped + y2_clipped) / 2 - y) / h
                    new_bw = (x2_clipped - x1_clipped) / w
                    new_bh = (y2_clipped - y1_clipped) / h

                    # Save only valid annotations
                    if new_bw > 0 and new_bh > 0:
                        tile_labels.append(f"{int(cls)} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}")

            # Save only tiles with annotations
            if tile_labels:
                tile = image[y:y + h, x:x + w]

                tile_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.png"
                cv2.imwrite(os.path.join(output_dir, tile_name), tile)

                with open(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{tile_id}.txt"), 'w') as f:
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

    # Create mosaic for tiled images
    tile_files = sorted([f for f in os.listdir(output_dir) if f.startswith(image_base_name) and f.endswith(".png")],
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))

    grid_cols = int(np.ceil(np.sqrt(len(tile_files))))
    grid_rows = int(np.ceil(len(tile_files) / grid_cols))

    max_height = max(cv2.imread(os.path.join(output_dir, f)).shape[0] for f in tile_files)
    max_width = max(cv2.imread(os.path.join(output_dir, f)).shape[1] for f in tile_files)

    tile_images = []
    for i in range(grid_rows * grid_cols):
        if i < len(tile_files):
            tile_file = tile_files[i]
            tile_image = cv2.imread(os.path.join(output_dir, tile_file))
            tile_label_file = os.path.join(output_dir, f"{os.path.splitext(tile_file)[0]}.txt")

            if os.path.exists(tile_label_file):
                with open(tile_label_file, 'r') as f:
                    for line in f:
                        cls, x_center, y_center, width, height = map(float, line.strip().split())
                        h, w, _ = tile_image.shape
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)

                        cv2.rectangle(tile_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(tile_image, str(int(cls)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            padded_tile = cv2.copyMakeBorder(
                tile_image,
                0, max_height - tile_image.shape[0],
                0, max_width - tile_image.shape[1],
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
        else:
            padded_tile = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        tile_images.append(padded_tile)

    mosaic_rows = [np.hstack(tile_images[i * grid_cols:(i + 1) * grid_cols]) for i in range(grid_rows)]
    mosaic = np.vstack(mosaic_rows)

    screen_width = 1920
    screen_height = 1080
    half_screen_width = screen_width // 2

    resized_image = resize_to_fit(image, half_screen_width, screen_height)
    resized_mosaic = resize_to_fit(mosaic, half_screen_width, screen_height)

    cv2.imshow(f"Original Annotations for {image_base_name}", resized_image)
    cv2.imshow(f"Tiled Mosaic for {image_base_name}", resized_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    base_dir = "/home/joakim/data/challenge"
    input_dir = os.path.join(base_dir, "test")
    output_dir = os.path.join(base_dir, "test_tiled")
    tile_size = (640, 640)
    overlap_fraction = 0.2
    img = "drone1"
    process_folder(input_dir, output_dir, tile_size, overlap_fraction)
    visualize_annotations(img, input_dir, output_dir)
