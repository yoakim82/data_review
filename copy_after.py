import os
import shutil
from datetime import datetime

def copy_files(source_folder, destination_folder, cutoff_time):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for foldername, subfolders, filenames in os.walk(source_folder):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)

            # Get file creation time
            creation_time = os.path.getctime(file_path)

            # Convert creation time to datetime object
            creation_datetime = datetime.fromtimestamp(creation_time)

            # Check if the file is newer than the cutoff time
            if creation_datetime > cutoff_time:
                # Recreate the folder structure in the destination folder
                relative_path = os.path.relpath(file_path, source_folder)
                destination_path = os.path.join(destination_folder, relative_path)

                # Ensure the destination folder for the file exists
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)

                # Copy the file to the destination folder
                shutil.copy2(file_path, destination_path)
                print(f"Copied {file_path} to {destination_path}")

if __name__ == "__main__":
    source_folder = 'seg_small_nerf_batches'
    destination_folder = '/media/joakim/T7/viser/copy/seg_small_nerf_batches'

    # Set the cutoff time (e.g., files created after January 1, 2023)
    cutoff_time = datetime(2023, 11, 16)

    copy_files(source_folder, destination_folder, cutoff_time)
