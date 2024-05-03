import os
import json
import numpy as np
import matplotlib.pyplot as plt
import statistics
import argparse

available_classes = {"multirotor": 29,
                     "fixedwing": 30,
                     "airliner": 31,
                     "bird": 32
                     }

out_classes = {"multirotor": 4,
               "fixedwing": 5,
               "airliner": 6,
               "bird": 7
               }
# for convenience, reverse the dict to use with printouts
class_names = {v: k for k, v in out_classes.items()}

def parse_json_files_from_list(file_list_path):
    # Read the list of folder paths from the file
    with open(file_list_path, 'r') as file:
        folder_paths = [line.strip() for line in file.readlines()]

    json_data_list = []

    # Iterate through each folder path and parse JSON files
    for folder_path in folder_paths:
        json_data_list.extend(parse_json_files(folder_path))

    return json_data_list


def load_yolov8_annotations_from_list(file_list_path):
    # Read the list of folder paths from the file
    with open(file_list_path, 'r') as file:
        folder_paths = [line.strip() for line in file.readlines()]

    annotation_data_list = []

    # Iterate through each folder path and load YOLOv8 annotations
    for folder_path in folder_paths:
        annotation_data_list.extend(load_yolov8_annotations(folder_path))

    return annotation_data_list


# Function to parse JSON files in the specified folder
def parse_json_files(folder_path):
    json_data_list = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # Open and read the JSON file
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                json_data_list.append(json_data)

    return json_data_list


def load_yolov8_annotations(folder_path):
    annotation_data_list = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                bboxes = []
                vehicle_class = []

                for line in lines:
                    # YOLOv8 format: class x_center y_center width height
                    data = line.strip().split()
                    class_label = int(data[0])
                    x_center, y_center, width, height = map(float, data[1:])

                    # Convert YOLO format to regular format (xmin, ymin, xmax, ymax)
                    xmin = int((x_center - width / 2) * 1920)  # Assuming image width is 1920
                    ymin = int((y_center - height / 2) * 1080)  # Assuming image height is 1080
                    xmax = int((x_center + width / 2) * 1920)
                    ymax = int((y_center + height / 2) * 1080)

                    bboxes.append([[xmin, ymin], [xmax, ymax]])

                    vehicle_class.append(class_label)

                metadata = {
                    "cloudiness": 0.0,
                    "precipitation": 0.0,
                    "precipitation_deposits": 0.0,
                    "wind_intensity": 0.0,
                    "sun_azimuth_angle": 0.0,
                    "sun_altitude_angle": 0.0,
                    "fog_density": 0.0,
                    "fog_distance": 0,
                    "wetness": 0.0
                }

                removed_bboxes = []
                removed_vehicle_class = []

                annotation_data = {
                    "bboxes": bboxes,
                    "metadata": metadata,
                    "vehicle_class": vehicle_class,  # Assuming vehicle class is not available in YOLOv8 format
                    "removed_bboxes": removed_bboxes,
                    "removed_vehicle_class": removed_vehicle_class
                }

                annotation_data_list.append(annotation_data)

    return annotation_data_list


def calculate_average_size_and_count(json_data_list):
    class_statistics = {}

    # Iterate through each parsed JSON
    for json_data in json_data_list:
        # Extract class information and bounding boxes
        vehicle_class = json_data.get('vehicle_class', [])
        removed_vehicle_class = json_data.get('removed_vehicle_class', [])
        bboxes = json_data.get('bboxes', [])
        removed_bboxes = json_data.get('removed_bboxes', [])

        # Combine both regular and removed bounding boxes
        all_bboxes = bboxes + removed_bboxes
        all_vehicle_class = vehicle_class + removed_vehicle_class

        # Iterate through each bounding box
        for class_label, bbox in zip(all_vehicle_class, all_bboxes):
            # Calculate the width and height of the bounding box
            width = bbox[1][0] - bbox[0][0]
            height = bbox[1][1] - bbox[0][1]

            # Calculate the area of the bounding box
            area = width * height

            # Update class statistics
            if class_label not in class_statistics:
                class_statistics[class_label] = {
                    'regular': {'total_width': 0, 'total_height': 0, 'total_area': 0, 'bbox_count': 0, 'areas': []},
                    'removed': {'total_width': 0, 'total_height': 0, 'total_area': 0, 'bbox_count': 0, 'areas': []},
                    'overall': {'total_width': 0, 'total_height': 0, 'total_area': 0, 'bbox_count': 0, 'areas': []}
                }

            class_statistics[class_label]['overall']['total_width'] += width
            class_statistics[class_label]['overall']['total_height'] += height
            class_statistics[class_label]['overall']['total_area'] += area
            class_statistics[class_label]['overall']['bbox_count'] += 1
            class_statistics[class_label]['overall']['areas'].append(area)

            if class_label in vehicle_class:
                class_statistics[class_label]['regular']['total_width'] += width
                class_statistics[class_label]['regular']['total_height'] += height
                class_statistics[class_label]['regular']['total_area'] += area
                class_statistics[class_label]['regular']['bbox_count'] += 1
                class_statistics[class_label]['regular']['areas'].append(area)
            elif class_label in removed_vehicle_class:
                class_statistics[class_label]['removed']['total_width'] += width
                class_statistics[class_label]['removed']['total_height'] += height
                class_statistics[class_label]['removed']['total_area'] += area
                class_statistics[class_label]['removed']['bbox_count'] += 1
                class_statistics[class_label]['removed']['areas'].append(area)

    # Print class statistics
    for class_label, stats in class_statistics.items():
        print(f"Class {class_label}:")
        print(
            f"  Regular Bboxes - Average Size: {stats['regular']['total_width'] / stats['regular']['bbox_count'] if stats['regular']['bbox_count'] > 0 else 0} (width) x {stats['regular']['total_height'] / stats['regular']['bbox_count'] if stats['regular']['bbox_count'] > 0 else 0} (height)")
        print(
            f"  Regular Bboxes - Average Area: {stats['regular']['total_area'] / stats['regular']['bbox_count'] if stats['regular']['bbox_count'] > 0 else 0}")
        print(
            f"  Regular Bboxes - Median Area: {statistics.median(stats['regular']['areas']) if stats['regular']['bbox_count'] > 0 else 0}")
        print(f"  Regular Bboxes - Number of Bounding Boxes: {stats['regular']['bbox_count']}")
        print("")
        print(
            f"  Removed Bboxes - Average Size: {stats['removed']['total_width'] / stats['removed']['bbox_count'] if stats['removed']['bbox_count'] > 0 else 0} (width) x {stats['removed']['total_height'] / stats['removed']['bbox_count'] if stats['removed']['bbox_count'] > 0 else 0} (height)")
        print(
            f"  Removed Bboxes - Average Area: {stats['removed']['total_area'] / stats['removed']['bbox_count'] if stats['removed']['bbox_count'] > 0 else 0}")
        print(
            f"  Removed Bboxes - Median Area: {statistics.median(stats['removed']['areas']) if stats['removed']['bbox_count'] > 0 else 0}")
        print(f"  Removed Bboxes - Number of Bounding Boxes: {stats['removed']['bbox_count']}")
        print("")
        print(
            f"  Overall - Average Size: {stats['overall']['total_width'] / stats['overall']['bbox_count'] if stats['overall']['bbox_count'] > 0 else 0} (width) x {stats['overall']['total_height'] / stats['overall']['bbox_count'] if stats['overall']['bbox_count'] > 0 else 0} (height)")
        print(
            f"  Overall - Average Area: {stats['overall']['total_area'] / stats['overall']['bbox_count'] if stats['overall']['bbox_count'] > 0 else 0}")
        print(
            f"  Overall - Median Area: {statistics.median(stats['overall']['areas']) if stats['overall']['bbox_count'] > 0 else 0}")
        print(f"  Overall - Number of Bounding Boxes: {stats['overall']['bbox_count']}")
        print("")


def plot_bbox_area_distribution(json_data_list, vehicle_class=None, bbox_type='all', n_bins=40, outfilename=""):
    areas = []

    for json_data in json_data_list:
        if bbox_type == 'all':
            bboxes = json_data.get('bboxes', []) + json_data.get('removed_bboxes', [])
        elif bbox_type == 'removed':
            bboxes = json_data.get('removed_bboxes', [])
        elif bbox_type == 'regular':
            bboxes = json_data.get('bboxes', [])
        else:
            raise ValueError("Invalid bbox_type. Choose 'all', 'removed', or 'regular'.")

        if vehicle_class is not None:
            bboxes = [bbox for bbox, v_class in zip(bboxes, json_data.get('vehicle_class', [])) if
                      v_class == vehicle_class]

        for bbox in bboxes:
            width = bbox[1][0] - bbox[0][0]
            height = bbox[1][1] - bbox[0][1]
            area = width * height
            areas.append(area)

    # Plot the distribution with a logarithmic x-axis
    plt.figure(figsize=(12, 3))
    plt.hist(areas, bins=n_bins, color='blue', alpha=0.7, log=True)
    className = class_names[vehicle_class] if vehicle_class is not None else "All"
    title = f'Distribution of Bounding Box Areas - {className} - Type {bbox_type.capitalize()}'
    plt.title(title)
    plt.xlabel('Bounding Box Area (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    # Save the plot as PNG
    save_plot_as_png(plt, f"{outfilename}_bbox_area_distribution_{className}_type_{bbox_type}")


def plot_metadata_distribution(json_data_list, outfilename=""):
    #metadata_params = ['cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
    #                   'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance', 'wetness']
    metadata_params = ['precipitation', 'precipitation_deposits', 'sun_azimuth_angle',
                       'sun_altitude_angle', 'cloudiness', 'fog_density']

    num_params = len(metadata_params)
    num_rows = (num_params + 1) // 2  # Adjust the number of rows based on the number of parameters

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))

    for i, param in enumerate(metadata_params):
        values = [json_data['metadata'][param] for json_data in json_data_list]

        row = i // 2
        col = i % 2

        axes[row, col].hist(values, bins=20, color='blue', alpha=0.7)
        axes[row, col].set_title(f'Distribution of {param}')
        axes[row, col].set_xlabel(param)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    # plt.show()
    # Save the plot as PNG
    save_plot_as_png(plt, f"{outfilename}_metadata_distribution")


def plot_bbox_positions(json_data_list, vehicle_class=None, bbox_type='all', outfilename=""):
    plt.figure(figsize=(10, 8))

    for json_data in json_data_list:
        if bbox_type == 'all':
            bboxes = json_data.get('bboxes', []) + json_data.get('removed_bboxes', [])
            classes = json_data.get('vehicle_class', []) + json_data.get('removed_vehicle_class', [])
        elif bbox_type == 'removed':
            bboxes = json_data.get('removed_bboxes', [])
            classes = json_data.get('removed_vehicle_class', [])
        elif bbox_type == 'regular':
            bboxes = json_data.get('bboxes', [])
            classes = json_data.get('vehicle_class', [])
        else:
            raise ValueError("Invalid bbox_type. Choose 'all', 'removed', or 'regular'.")

        if vehicle_class is not None:
            # Filter bounding boxes based on the specified vehicle_class
            bboxes = [bbox for bbox, v_class in zip(bboxes, classes) if v_class == vehicle_class]

        for bbox in bboxes:
            x = [bbox[0][0], bbox[1][0], bbox[1][0], bbox[0][0], bbox[0][0]]
            y = [bbox[0][1], bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1]]

            plt.plot(x, y, color='blue', alpha=0.2)

    className = class_names[vehicle_class] if vehicle_class is not None else "All"
    title = f'Bounding Box Positions (CDF) {className} - Type {bbox_type.capitalize()}'
    plt.title(title)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    # plt.show()
    # Save the plot as PNG
    save_plot_as_png(plt, f"{outfilename}_bbox_positions_{className}_type_{bbox_type}")


def plot_bbox_areas(json_data_list, bbox_type='all', vehicle_class=None):
    # Find the maximum x and y coordinates to determine the size of the matrix
    max_x = max(max(bbox[1][0] for bbox in json_data['bboxes']) for json_data in json_data_list)
    max_y = max(max(bbox[1][1] for bbox in json_data['bboxes']) for json_data in json_data_list)

    max_y = 1080
    max_x = 1920

    # Create an empty matrix with zeros
    matrix = np.zeros((int(max_y) + 1, int(max_x) + 1))

    # Iterate through each JSON data and update the matrix for the selected type of bounding boxes and vehicle class
    for json_data in json_data_list:
        if bbox_type == 'all':
            if vehicle_class is not None:
                bboxes = [bbox for bbox, v_class in zip(json_data.get('bboxes', []), json_data.get('vehicle_class', []))
                          if v_class == vehicle_class]
            else:
                bboxes = json_data.get('bboxes', [])
        elif bbox_type == 'removed':
            if vehicle_class is not None:
                bboxes = [bbox for bbox, v_class in
                          zip(json_data.get('removed_bboxes', []), json_data.get('removed_vehicle_class', [])) if
                          v_class == vehicle_class]
            else:
                bboxes = json_data.get('removed_bboxes', [])
        elif bbox_type == 'regular':
            if vehicle_class is not None:
                bboxes = [bbox for bbox, v_class in zip(json_data.get('bboxes', []), json_data.get('vehicle_class', []))
                          if v_class == vehicle_class]
            else:
                bboxes = json_data.get('bboxes', [])
        else:
            raise ValueError("Invalid bbox_type. Choose 'all', 'removed', or 'regular'.")

        for bbox in bboxes:
            x_start, y_start = map(int, bbox[0])
            x_end, y_end = map(int, bbox[1])

            # Increment the matrix values inside the bounding box by 1
            matrix[y_start:y_end, x_start:x_end] += 1

    # Normalize the matrix
    matrix /= matrix.max()

    # Plot the matrix
    plt.imshow(matrix, cmap='viridis', extent=[0, max_x, 0, max_y], origin='lower', alpha=0.8)
    className = class_names[vehicle_class] if vehicle_class is not None else "All"
    title = f'Bounding Box Areas ({bbox_type.capitalize()} - {className})'
    plt.title(title)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.colorbar()
    # plt.show()
    # Save the plot as PNG
    save_plot_as_png(plt, f"bbox_areas_{className}_type_{bbox_type}")


def plot_bbox_heatmap(json_data_list, bins=40):
    plt.figure(figsize=(10, 8))

    all_x = []
    all_y = []

    for json_data in json_data_list:
        bboxes = json_data.get('bboxes', [])

        for bbox in bboxes:
            x = np.linspace(bbox[0][0], bbox[1][0], num=bins)
            y = np.linspace(bbox[0][1], bbox[1][1], num=bins)

            all_x.extend(x)
            all_y.extend(y)

    plt.hist2d(all_x, all_y, bins=bins, cmap='viridis')
    plt.colorbar(label='Frequency')

    plt.title('Bounding Box Heatmap')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    # plt.show()
    # Save the plot as PNG
    save_plot_as_png(plt, "bbox_heatmap")


def print_all_classes_ids(json_data_list):
    all_classes = set()

    for json_data in json_data_list:
        vehicle_classes = json_data.get('vehicle_class', [])
        removed_classes = json_data.get('removed_vehicle_class', [])

        all_classes.update(vehicle_classes)
        all_classes.update(removed_classes)

    print("All Different Classes:")
    for class_label in sorted(all_classes):
        print(f"Class {class_label}")


# Global output path for saving figures
output_path = './boxstat/'

# Ensure the output path exists
os.makedirs(output_path, exist_ok=True)


def save_plot_as_png(figure, filename):
    output_file_path = os.path.join(output_path, filename + '.png')
    figure.savefig(output_file_path)
    print(f"Saved plot as PNG: {output_file_path}")


########### main code #############


# single paths
ANNOTATION_PATH = '/media/gionji/New Volume/saab_viser_carla_dataset/data_review/new_batches/world_0_drones_1/out_bbox'
yolov8_annotations_folder = '/content/drive/MyDrive/MDU/DatasetsStats/data/dvb_yolo_annotations'


def main(boxdirfile, outfilename=""):
    #parser = argparse.ArgumentParser(description='Process JSON and YOLO annotations.')
    #parser.add_argument('file_list_path', type=str, help='Path to the file containing a list of folder paths.')

    #args = parser.parse_args()

    if outfilename == "":
        outfilename = boxdirfile.split('boxstat_')[-1]

    # Parse JSON files in the specified annotation folder
    #boxdirfile = "boxstat_seg_small_batches.txt"
    parsed_data = parse_json_files_from_list(boxdirfile)

    # Parse JSON files in the specified annotation folder
    # parsed_data = parse_json_files(ANNOTATION_PATH)

    ### General stats
    # Print the bboxes stats
    # It prints the average width,height of bboxes. The average and median area.
    # It separates stats by all bboxes, regular (the bboxes present in the dataset) and removed.
    calculate_average_size_and_count(parsed_data)

    # It prints the simulation weather parameters distributions
    plot_metadata_distribution(parsed_data, outfilename=outfilename)

    ### It plots the bboxes area distribution
    ##It can be filtered by class id and Presence(all, regular, removed)

    # here we compare the regular areas
    bins = 160
    # multirotor
    plot_bbox_area_distribution(parsed_data, vehicle_class=out_classes['multirotor'], bbox_type='regular', n_bins=bins, outfilename=outfilename)
    # fixedwing
    plot_bbox_area_distribution(parsed_data, vehicle_class=out_classes['fixedwing'], bbox_type='regular', n_bins=bins, outfilename=outfilename)
    # bird
    plot_bbox_area_distribution(parsed_data, vehicle_class=out_classes['bird'], bbox_type='regular', n_bins=bins, outfilename=outfilename)

    # Bboxes areas localization
    #plot_bbox_positions(parsed_data, vehicle_class=out_classes['bird'], bbox_type='regular')
    #plot_bbox_positions(parsed_data, vehicle_class=out_classes['bird'], bbox_type='removed')
    #plot_bbox_positions(parsed_data, vehicle_class=out_classes['bird'], bbox_type='all')

    plot_bbox_positions(parsed_data, vehicle_class=out_classes['multirotor'], bbox_type='regular', outfilename=outfilename)
    plot_bbox_positions(parsed_data, vehicle_class=out_classes['fixedwing'], bbox_type='regular', outfilename=outfilename)
    plot_bbox_positions(parsed_data, vehicle_class=out_classes['bird'], bbox_type='regular', outfilename=outfilename)
    plot_bbox_positions(parsed_data, vehicle_class=out_classes['airliner'], bbox_type='regular', outfilename=outfilename)


if __name__ == "__main__":

    main(boxdirfile="boxstat_seg_small_batches.txt", outfilename="synth_only")
    main(boxdirfile="boxstat_swan.txt", outfilename="swan-perturb")
    main(boxdirfile="boxstat_all.txt", outfilename="all")




