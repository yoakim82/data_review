import os
from moviepy import *

def get_video_resolutions(folder_path):
    with open("resolutions.txt", "w") as output_file:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Check for video file extensions
                file_path = os.path.join(folder_path, filename)
                try:
                    video = VideoFileClip(file_path)
                    width, height = video.size
                    output_file.write(f"{filename},{width},{height}\n")
                    video.close()
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

def get_video_metrics(folder_path, filename):


    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        get_video_resolutions(folder_path)
        print("Resolutions saved in 'resolutions.txt'")
    else:
        print("Invalid folder path. Please provide a valid path to a directory containing video files.")


#get_video_metrics(folder_path, filename)

def extract_by_width(file_path, specific_width):
    matching_entries = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            #entry = line.strip().split(',')
            filename, width, height = line.strip().split(',')
            #change filename from video file to annotation file
            new_filename = '.'.join(filename.split('.')[:-1]) + '.txt'
            #width, height = resolution.split('x')
            if int(width) == specific_width:
                matching_entries.append(new_filename)

    return matching_entries


folder_path = "/home/joakim/data_review/drone-vs-birds/train_videos/"
filename = "resolutions.txt"

videos_4k = extract_by_width(filename, 3840)
videos_hd = extract_by_width(filename, 1920)

print(f"4K videos:")
print(videos_4k)
