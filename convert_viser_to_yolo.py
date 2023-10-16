import os
import random
import argparse
import json
from glob import glob

def create_yolo_labels_folder(folder):
    labels_folder = os.path.join(folder, "")
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    return labels_folder

def convert_to_yolo_format(bboxes, image_width, image_height):
    yolo_bboxes = []
    for bbox in bboxes:
        x_center = (bbox[0][0] + bbox[1][0]) / (2 * image_width)
        y_center = (bbox[0][1] + bbox[1][1]) / (2 * image_height)
        width = (bbox[1][0] - bbox[0][0]) / image_width
        height = (bbox[1][1] - bbox[0][1]) / image_height
        yolo_bboxes.append([x_center, y_center, width, height])
    return yolo_bboxes

def convert_annotation(annotation_file, labels_folder, image_path):
    with open(annotation_file, "r") as f:
        annotation_data = json.load(f)

    image_width = 1920  # Update with the actual image width
    image_height = 1080  # Update with the actual image height

    yolo_bboxes = convert_to_yolo_format(annotation_data.get("bboxes", []), image_width, image_height)

    label_filename = os.path.join(labels_folder, os.path.basename(annotation_file).replace(".txt", ".txt"))
    with open(label_filename, "w") as label_file:
        for i, bbox in enumerate(yolo_bboxes):
            # assign the class labels according to this lookup table:

            class_definition = {4: 0,  # multirotor
                                5: 1,  # fixedwing
                                6: 2,  # airliner
                                7: 3  # bird
                                }
            class_label = class_definition[annotation_data['vehicle_class'][i]]

            label_file.write(f"{class_label} {' '.join(map(str, bbox))}\n")

def split_dataset(out_rgb_folder, train_ratio, test_ratio, valid_ratio):
    image_paths = glob(os.path.join(out_rgb_folder, "*.png"))
    random.shuffle(image_paths)

    total_images = len(image_paths)
    train_count = int(total_images * train_ratio)
    test_count = int(total_images * test_ratio)
    valid_count = total_images - train_count - test_count

    train_paths = image_paths[:train_count]
    test_paths = image_paths[train_count:train_count + test_count]
    valid_paths = image_paths[train_count + test_count:]

    with open(os.path.join(out_rgb_folder, "train.txt"), "w") as train_file:
        for path in train_paths:
            train_file.write(f"{os.path.abspath(path)}\n")

    with open(os.path.join(out_rgb_folder, "test.txt"), "w") as test_file:
        for path in test_paths:
            test_file.write(f"{os.path.abspath(path)}\n")

    with open(os.path.join(out_rgb_folder, "valid.txt"), "w") as valid_file:
        for path in valid_paths:
            valid_file.write(f"{os.path.abspath(path)}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert annotation files to YOLO format and split the dataset.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing 'out_rgb' and 'out_bbox'.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train set ratio (default: 0.7).")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio (default: 0.2).")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1).")
    args = parser.parse_args()

    out_rgb_folder = os.path.join(args.folder, "out_rgb")
    labels_folder = create_yolo_labels_folder(out_rgb_folder)

    out_bbox_folder = os.path.join(args.folder, "out_bbox")
    annotation_files = glob(os.path.join(out_bbox_folder, "*.txt"))

    for annotation_file in annotation_files:
        convert_annotation(annotation_file, labels_folder, out_rgb_folder)

    #split_dataset(out_rgb_folder, args.train_ratio, args.test_ratio, args.valid_ratio)

if __name__ == "__main__":
    main()
