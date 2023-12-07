import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")

    # Loop through each frame
    for frame_num in range(frame_count):
        # Read the frame
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print(f"Error reading frame {frame_num}")
            break

        # Define the output image file path
        output_path = os.path.join(output_folder, f"img_{frame_num}.png")

        # Save the frame as PNG image
        cv2.imwrite(output_path, frame)

        # Print progress
        print(f"Frame {frame_num}/{frame_count} saved: {output_path}")

    # Release the video capture object
    cap.release()

    print("Frames extraction completed.")

if __name__ == "__main__":
    # Specify the input video file and output folder
    input_video_path = "/home/lxc/stjoly/datasets/viser/swan/VTOL_151624.mp4"
    output_folder = "/home/lxc/stjoly/datasets/viser/swan/"

    # Call the function to extract frames
    extract_frames(input_video_path, output_folder)
