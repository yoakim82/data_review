import os
import pdb
import random
import argparse
import json
from glob import glob
import yaml
import cv2
from tqdm import tqdm


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


def partition_dataset(dataset_folder, train_folders, validation_folders, test_folders, background_folders, background_ratio, exp_name):
    train_paths = []
    val_paths = []
    test_paths = []
    background_paths = []

    for folder in train_folders:
        train_paths.append(glob(os.path.join(dataset_folder, folder, "out_rgb", "*.png")))
    for folder in validation_folders:
        val_paths.append(glob(os.path.join(dataset_folder, folder, "out_rgb", "*.png")))
    for folder in test_folders:
        test_paths.append(glob(os.path.join(dataset_folder, folder, "out_rgb", "*.png")))

    total_trains = sum([len(i) for i in train_paths])
    wanted_total_num_background = total_trains * background_ratio
    wanted_num_background_per_folder = int(wanted_total_num_background / len(background_folders))
    for folder in background_folders:
        image_paths = glob(os.path.join(dataset_folder, folder, "backgrounds", "out_rgb", "*.png"))
        random.shuffle(image_paths)
        background_paths.append(image_paths[:wanted_num_background_per_folder])

    total_valids = len(val_paths)
    total_tests = len(test_paths)


    with open(os.path.join(dataset_folder, f"train_{exp_name}.txt"), "w") as train_file:
        for folder in train_paths:
            for path in folder:
                train_file.write(f"{os.path.abspath(path)}\n")

        # add background images only to training portion
        for folder in background_paths:
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



def create_dataset_yaml(exp_name, folder, framework="yolov7"):

    yaml_file_name = f"{os.path.join(folder, exp_name)}_dataset.yaml"

    if framework=="yolov7":
        data = {
            'train': f"./{folder}/train_{exp_name}.txt",
            'val':   f"./{folder}/valid_{exp_name}.txt",
            'test':  f"./{folder}/test_{exp_name}.txt",
            'nc': 4,
            'names': ['multirotor', 'fixedwing', 'airliner', 'bird']
        }
        command = f"python train_aux.py --img 640 --batch 16 --epochs 10 --data {yaml_file_name} --cfg ./cfg/training/yolov7-w6.yaml --weights '' --name {exp_name}"

    elif framework=="yolov8":
        data = {
            'train': f"./train_{exp_name}.txt",
            'val': f"./valid_{exp_name}.txt",
            'test': f"./test_{exp_name}.txt",
            'nc': 4,
            'names': ['multirotor', 'fixedwing', 'airliner', 'bird']
        }
        command = f"python train.py --directory {folder} --img_size 1920 --batch 6 --epochs 100 --data {yaml_file_name} --name {exp_name}"

    with open(yaml_file_name, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    #command = f"python train_aux.py --img 640 --batch 16 --epochs 10 --data {yaml_file_name} --cfg ./cfg/training/yolov7-w6.yaml --weights '' --name {exp_name}"
    command_file_name = f"{os.path.join(folder, exp_name)}_train.sh"
    with open(command_file_name, "a") as script_file:
        script_file.write(command + "\n")


def find_matching_file(folder, pattern, extensions):
    for ext in extensions:
        gurka = os.path.join(folder, f"{pattern}.{ext}")
        files = glob(gurka)
        if files:
            return files[0]  # Return the first matching file found
    return None  # Return None if no matching file is found


def list_of_frames(file_path):

    detected_frames = []

    with open(file_path, "r") as text_file:
        lines = text_file.readlines()

    for i, line in enumerate(lines):
        values = line.split()
        has_object = int(values[1])
        detected_frames.append((i, line))

    return detected_frames


def filter_frames(lines_w_detections, num_frames_to_save):

    random.shuffle(lines_w_detections)

    selected_lines = lines_w_detections[0:num_frames_to_save]

    sorted_lines = sorted(selected_lines, key=lambda x: x[0])
    return sorted_lines


def get_frame(cap, frame_number):
    # Set the video capture object to the desired frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    return cap.read()


def convert_drone_vs_bird_to_yolo(data_folder, annotation_folder, filename, output_dir, num_frames_to_save):
    # Define input file paths
    print(f"processing file {filename}, extracting {num_frames_to_save} random images.")
    possible_extensions = ["avi", "mpg", "mp4", "AVI", "MPG", "MP4"]
    video_file_path = find_matching_file(data_folder, filename, possible_extensions)
    text_file_path = os.path.join(annotation_folder, f"{filename}.txt")

    data = []
    # Create an output directory to store images and annotations

    os.makedirs(output_dir, exist_ok=True)

    # Open the text file to read detection data
    #with open(text_file_path, "r") as text_file:
    #    lines = text_file.readlines()

    lines_w_detections = list_of_frames(text_file_path)
    selected_detections = filter_frames(lines_w_detections, num_frames_to_save)

    # Open the video file for frame extraction
    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    frame_number = 0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}, size of frames: {frame_width}, {frame_height}")

    #ret, frame = get_frame(cap, frame_number)

    for (frame_number, line) in tqdm(selected_detections):
        values = line.split()
        has_object = int(values[1])

        # always read the frame from the video (to stay in sync)
        ret, frame = get_frame(cap, frame_number)
        if not ret:
            print(f"reached end of file, frame number = {frame_number}.")
            break
        #print(f"processing frame {frame_number}")
        if has_object >= 1:
            boxes = []
            classes = []
            for i in range(has_object):
                x, y, w, h = map(int, values[2+i*5:6+i*5])
                class_name = values[6+5*i].strip()

                box = [[x, y], [x+w, y+h]]
                boxes.append(box)
                classes.append(class_name)

            yolo_bboxes = convert_to_yolo_format(bboxes=boxes, image_width=frame_width, image_height=frame_height)

            # Save the frame as an image
            fname = f"{filename}_fr{frame_number}"
            data.append([classes, yolo_bboxes, fname, frame])

    #random.shuffle(data)
    #selected_data = data[:num_frames_to_save]

    # for convenience, unpack data record after shuffling
    for (classes, yolo_bboxes, filename, frame) in tqdm(data):

        frame_path = os.path.join(output_dir, f"{filename}.png")
        cv2.imwrite(frame_path, frame)

        annotation_path = os.path.join(output_dir, f"{filename}.txt")
        with open(annotation_path, "w") as label_file:
            for i, bbox in enumerate(yolo_bboxes):
                # assign the class labels according to this lookup table:

                class_definition = {"drone": 0,  # multirotor
                                    "fixedwing": 1,  # fixedwing
                                    "airliner": 2,  # airliner
                                    "bird": 3  # bird
                                    }
                class_label = class_definition[classes[i]]

                label_file.write(f"{class_label} {' '.join(map(str, bbox))}\n")


    # Release the video capture object and clean up
    cap.release()
    cv2.destroyAllWindows()


def setup_dataset(dataset_folder, test_folders, exp_name):
    test_paths = []

    for folder in test_folders:
        test_paths.append(glob(os.path.join(dataset_folder, folder, "*.png")))

    total_tests = len(test_paths)

    with open(os.path.join(dataset_folder, f"test_{exp_name}.txt"), "w") as test_file:
        for folder in test_paths:
            for path in folder:
                test_file.write(f"{os.path.abspath(path)}\n")


def count_frames_w_annotations(annotation_folder):
    annotations = []
    for filename in glob(os.path.join(annotation_folder, "*.txt")):
        with open(filename, "r") as text_file:
            lines = text_file.readlines()
        count = 0
        for line in lines:
            values = line.split()
            has_object = int(values[1])
            if has_object > 0:
                count+= 1

        annotations.append(count)
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Convert annotation files to YOLO format and split the dataset.")
    parser.add_argument("--folder", required=True, help="Path to the folder containing 'out_rgb' and 'out_bbox'.")
    parser.add_argument("--train_num", type=int, default=5, help="Number of training folders (default: 5).")
    parser.add_argument("--valid_num", type=int, default=2, help="Number of validation folders (default: 2).")
    parser.add_argument("--test_num,", type=int, default=1, help="Number of test folders (default: 1).")
    parser.add_argument("--experiment_name", type=str, default="exp", help="Name of experiment (default: exp)")
    parser.add_argument("--shift", type=int, default=0, help="how many steps to rotate data folders before splitting (default: 0)")
    parser.add_argument("--source", type=str, default="carla", help="use 'carla' if input is synthetic data or 'drones-vs-birds' for real test data (default: 'carla')")

    args = parser.parse_args()

    if args.source == "carla":
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

    elif args.source == "drone-vs-birds":
        save_ratio = 0.03
        exp_name = f"drone_vs_birds_{args.shift}"
        #partition_dataset(args.folder, train_list, val_list, test_list, exp_name)

        data_folder = os.path.join(args.folder, "train_videos")
        annotation_folder = os.path.join(args.folder, "annotations")
        num_drone_frames = count_frames_w_annotations(annotation_folder)
        num_frames_to_save = [int(x * save_ratio) for x in num_drone_frames]
        output_dir = os.path.join(args.folder, "yolo_annotations")

        startFromHere = False
        print("Extracting frames from video sequences and converting to yolo format.")
        for (num_frames, filename) in zip(num_frames_to_save, glob(os.path.join(annotation_folder, "*.txt"))):
            fname = os.path.basename(filename).split(".")[0]

            #if os.path.basename(filename) == "2019_10_16_C0003_3633_inspire.txt":
            #    startFromHere = True

            if startFromHere:
                convert_drone_vs_bird_to_yolo(data_folder=data_folder,
                                              filename=fname,
                                              annotation_folder=annotation_folder,
                                              output_dir=output_dir,
                                              num_frames_to_save=num_frames)

            file_paths = glob(os.path.join(output_dir, "*.png"))
            with open(os.path.join(args.folder, f"{exp_name}.txt"), "w") as test_file:
                for path in file_paths:
                    test_file.write(f"{os.path.abspath(path)}\n")
            #create_dataset_yaml(exp_name, args.folder)

if __name__ == "__main__":
    main()

