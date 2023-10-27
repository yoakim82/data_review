import os
import random
import argparse
import json
from glob import glob
import yaml
import cv2

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


def partition_dataset(dataset_folder, train_folders, validation_folders, test_folders, exp_name):
    train_paths = []
    val_paths = []
    test_paths = []

    for folder in train_folders:
        train_paths.append(glob(os.path.join(dataset_folder, folder, "out_rgb", "*.png")))
    for folder in validation_folders:
        val_paths.append(glob(os.path.join(dataset_folder, folder, "out_rgb", "*.png")))
    for folder in test_folders:
        test_paths.append(glob(os.path.join(dataset_folder, folder, "out_rgb", "*.png")))

    total_trains = len(train_paths)
    total_valids = len(val_paths)
    total_tests = len(test_paths)


    with open(os.path.join(dataset_folder, f"train_{exp_name}.txt"), "w") as train_file:
        for folder in train_paths:
            for path in folder:
                train_file.write(f"{os.path.abspath(path)}\n")


    with open(os.path.join(dataset_folder, f"valid_{exp_name}.txt"), "w") as valid_file:
        for folder in val_paths:
            for path in folder:
                valid_file.write(f"{os.path.abspath(path)}\n")

    with open(os.path.join(dataset_folder, f"test_{exp_name}.txt"), "w") as test_file:
        for folder in test_paths:
            for path in folder:
                test_file.write(f"{os.path.abspath(path)}\n")


def get_list_of_data_folders(folder):
    folders = [f"world_{n}_drones_1" for n in range(8)]
    return folders

def permute_list(list, shift=0):
    return list[shift:] + list[:shift]



def create_dataset_yaml(exp_name, folder):

    data = {
        'train': f"./{folder}/train_{exp_name}.txt",
        'val':   f"./{folder}/valid_{exp_name}.txt",
        'test':  f"./{folder}/test_{exp_name}.txt",
        'nc': 4,
        'names': ['multirotor', 'fixedwing', 'airliner', 'bird']
    }
    yaml_file_name = f"{os.path.join(folder, exp_name)}_dataset.yaml"
    with open(yaml_file_name, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    command = f"python train_aux.py --img 640 --batch 16 --epochs 10 --data {yaml_file_name} --cfg ./cfg/training/yolov7-w6.yaml --weights '' --name {exp_name}"
    command_file_name = f"{os.path.join(folder, exp_name)}_train.sh"
    with open(command_file_name, "a") as script_file:
        script_file.write(command + "\n")


def convert_drone_vs_bird_to_yolo(data_folder, annotation_folder, filename, output_dir):
    # Define input file paths
    video_file_path = os.path.join(data_folder, f"{filename}.mp4")
    text_file_path = os.path.join(annotation_folder, f"{filename}.txt")

    # Create an output directory to store images and annotations

    os.makedirs(output_dir, exist_ok=True)

    # Open the text file to read detection data
    with open(text_file_path, "r") as text_file:
        lines = text_file.readlines()

    # Open the video file for frame extraction
    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    frame_number = 0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    for line in lines:
        values = line.split()
        has_object = int(values[1])

        # always read the frame from the video (to stay in sync)
        ret, frame = cap.read()
        if not ret:
            print(f"reached end of file, frame number = {frame_number}.")
            break

        if has_object == 1:
            x, y, w, h = map(int, values[2:6])
            class_name = values[6].strip()

            # Read the frame from the video


            # Save the frame as an image
            frame_filename = f"{filename}_fr{frame_number}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            box = [[x, y], [x+w, y+h]]
            yolo_bboxes = convert_to_yolo_format(bboxes=[box], image_width=frame_width, image_height=frame_height)

            #label_filename = os.path.join(labels_folder, os.path.basename(annotation_file).replace(".txt", ".txt"))
            annotation_filename = os.path.join(output_dir, f"{filename}_fr{frame_number}.txt")
            with open(annotation_filename, "w") as label_file:
                for i, bbox in enumerate(yolo_bboxes):
                    # assign the class labels according to this lookup table:

                    class_definition = {"drone": 0,  # multirotor
                                        "5": 1,  # fixedwing
                                        "6": 2,  # airliner
                                        "7": 3  # bird
                                        }
                    class_label = class_definition[class_name]

                    label_file.write(f"{class_label} {' '.join(map(str, bbox))}\n")

        frame_number += 1

    # Release the video capture object and clean up
    cap.release()
    cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(description="Convert annotation files to YOLO format and split the dataset.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing 'out_rgb' and 'out_bbox'.")
    parser.add_argument("--train_num", type=int, default=5, help="Number of training folders (default: 5).")
    parser.add_argument("--test_num,", type=int, default=1, help="Number of test folders (default: 1).")
    parser.add_argument("--valid_num", type=int, default=2, help="Number of validation folders (default: 2).")
    parser.add_argument("--experiment_name", default="test")
    parser.add_argument("--shift", type=int, default=0, help="how many steps to rotate data folders before splitting (default: 0)")

    args = parser.parse_args()
    dlist = get_list_of_data_folders([])
    shifted_list = permute_list(dlist, args.shift)

    for folder in shifted_list:

        out_rgb_folder = os.path.join(args.folder, folder, "out_rgb")
        labels_folder = create_yolo_labels_folder(out_rgb_folder)

        out_bbox_folder = os.path.join(args.folder, folder, "out_bbox")
        annotation_files = glob(os.path.join(out_bbox_folder, "*.txt"))

        for annotation_file in annotation_files:
            convert_annotation(annotation_file, labels_folder, out_rgb_folder)

    #if i < args.train_num:
    #elif < i (args.train_num+args.val_num):
    #split_dataset(out_rgb_folder, args.train_ratio, args.test_ratio, args.valid_ratio)
    train_list = shifted_list[0:args.train_num]
    val_list = shifted_list[args.train_num:args.train_num+args.valid_num]
    test_list = shifted_list[args.train_num+args.valid_num:]
    exp_name = f"exp_{args.shift}"
    partition_dataset(args.folder, train_list, val_list, test_list, exp_name)
    create_dataset_yaml(exp_name, args.folder)


if __name__ == "__main__":
    #main()

    convert_drone_vs_bird_to_yolo(data_folder="drone-vs-birds",
                                  annotation_folder=os.path.join("drone-vs-birds", "annotations"),
                                  filename="00_01_52_to_00_01_58", output_dir=os.path.join("drone-vs-birds", "yolo_annotations"))

