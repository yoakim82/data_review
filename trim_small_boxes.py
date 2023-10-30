import glob
import json
import os

import matplotlib.pyplot as plt


def append_to_script(script_file_name, command):
    with open(script_file_name, "a") as script_file:
        script_file.write(command + "\n")

def process_json(json_file, area_threshold, box_areas, removed_box_areas, trim_script):
    with open(json_file, 'r') as f:
        data = json.load(f)

    bboxes = data["bboxes"]
    vehicle_class = data["vehicle_class"]
    removed_bboxes = data["removed_bboxes"]
    removed_vehicle_class = data["removed_vehicle_class"]

    i, j = 0, 0
    while j < len(removed_bboxes):
        rbox = removed_bboxes[j]
        min_x, min_y = rbox[0]
        max_x, max_y = rbox[1]
        area = (max_x - min_x) * (max_y - min_y)

        #print(f"INFO: area of removed box {j} is {area} pixels.")
        j += 1

        removed_box_areas.append(area)


    while i < len(bboxes):
        box = bboxes[i]
        min_x, min_y = box[0]
        max_x, max_y = box[1]
        area = (max_x - min_x) * (max_y - min_y)

        if area < area_threshold:
            print(f"area of box {i} is {area} pixels: too small, removing it.")
            removed_bboxes.append(bboxes.pop(i))
            removed_vehicle_class.append(vehicle_class.pop(i))
        else:
            print(f"box {i} with area {area} pixels is kept.")

        i += 1

        box_areas.append(area)

    if len(bboxes) == 0:
        file_name = os.path.basename(json_file)
        append_to_script(script_file_name=trim_script, command=f"find . -name \x5c{file_name.split('.')[0]}.* -type f -delete")

    if area_threshold > 0:
        data["bboxes"] = bboxes
        data["vehicle_class"] = vehicle_class
        data["removed_bboxes"] = removed_bboxes
        data["removed_vehicle_class"] = removed_vehicle_class

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)


def process_files(pattern, area_threshold, scriptname):
    file_list = glob.glob(pattern)
    box_areas = []
    removed_box_areas = []
    trim_script = scriptname

    for file in file_list:
        print(f"processing file {file}:")
        process_json(file, area_threshold, box_areas, removed_box_areas, trim_script)

    return box_areas, removed_box_areas

# Example usage:
#json_file = "json_test.txt"
#area_threshold = 100  # You can set your own threshold here
#process_json(json_file, area_threshold)


# Example usage:
folder = "new_batches"
patterns = [f"{folder}/world_{i}_drones_1/out_bbox/*.txt" for i in range(8)]

#pattern = "json*.txt"
area_threshold = 100  # remove annotation of any boxes greater than 100 pixels
for i, pattern in enumerate(patterns):

    box_areas, removed_box_areas = process_files(pattern, area_threshold, f"trimmed_images_world_{i}_drones_1.sh")

    plt.hist(box_areas, bins=50, range=(0, 1000))
    plt.title(f"Box Area Histogram for world_{i}_drones_1")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    plt.savefig(f"area_hist_world_{i}_drones_1.png")

#plt.hist(removed_box_areas, bins=50)
#plt.title("Removed Box Area Histogram")
#plt.xlabel("Area")
#plt.ylabel("Frequency")
#plt.show()