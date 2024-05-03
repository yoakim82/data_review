import json
import os
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from collections import defaultdict

class_colors = {0: "red",
                1: "green",
                2: "cyan",
                3: "magenta"}


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x


class VideoAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW",
                           self.save_state_and_exit)  # Bind the function to the window closing event
        self.root.title("Video Annotation App")
        self.current_file = None
        self.current_frame = None
        self.current_class = None
        self.current_box = None
        self.annotation_boxes = []
        self.canvas = tk.Canvas(root, bg="gray", width=1920 - 250, height=(1920 - 250) * 1080 / 1920)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.cap = None
        self.init_left_panel()
        self.load_default_image()
        self.annotated_boxes = defaultdict(list)

        self.root.bind('<Left>', self.step_backward)
        self.root.bind('<Right>', self.step_forward)

    def delete_selected_item(self, event):
        selected_index = self.annotation_listbox.curselection()
        if selected_index:
            index = int(selected_index[0])
            self.annotation_listbox.delete(index)
            # Remove the deleted annotation from the list of annotations for the current frame
            if self.current_frame in self.annotation_boxes:
                del self.annotation_boxes[self.current_frame][index]
        self.update_annotation_listbox()

    def update_output_folder(self, event):
        self.output_folder = self.output_folder_entry.get()

    def init_left_panel(self):
        left_frame = tk.Frame(self.root, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        open_button = tk.Button(left_frame, text="Open File", command=self.open_file)
        open_button.pack()

        new_multirotor_button = tk.Button(left_frame, text="New Multirotor", command=lambda: self.set_class(0))
        new_multirotor_button.pack()

        new_fixedwing_button = tk.Button(left_frame, text="New Fixedwing", command=lambda: self.set_class(1))
        new_fixedwing_button.pack()

        new_airliner_button = tk.Button(left_frame, text="New Airliner", command=lambda: self.set_class(2))
        new_airliner_button.pack()

        new_bird_button = tk.Button(left_frame, text="New Bird", command=lambda: self.set_class(3))
        new_bird_button.pack()

        save_annotations_button = tk.Button(left_frame, text="Save Annotations", command=self.save_annotations)
        save_annotations_button.pack()

        self.annotation_listbox = tk.Listbox(left_frame)
        self.annotation_listbox.pack()

        self.annotation_listbox.bind("<Delete>", self.delete_selected_item)

        output_folder_label = tk.Label(left_frame, text="Output Folder:")
        output_folder_label.pack()

        self.output_folder_entry = tk.Entry(left_frame)
        self.output_folder_entry.pack()

        self.output_folder_entry.bind("<KeyRelease>", self.update_output_folder)
        self.root.update()

    def load_default_image(self):
        # Load a default image or show an empty canvas
        pass

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        self.open_stream(file_path)

    def open_stream(self, file_path):
        if file_path:
            self.current_file = file_path
            self.current_frame = 0

            # Open the video file using OpenCV
            self.cap = cv2.VideoCapture(file_path)
            self.seek_frame()
            self.width = self.frame.shape[1]
            self.height = self.frame.shape[0]

    def seek_frame(self):

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.update_frame()

    def update_frame(self):
        # Load the frame and convert to PhotoImage
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(frame)

        # Calculate scaling factors to fit the image to the canvas
        width, height = frame.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate the scaling factor to fit the image inside the canvas
        if canvas_width < width or canvas_height < height:
            # Image is larger than the canvas, scale down
            ratio_w = canvas_width / width
            ratio_h = canvas_height / height
            scale = min(ratio_w, ratio_h)
            width = int(width * scale)
            height = int(height * scale)

            frame = frame.resize((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(frame)
        else:
            self.canvas.config(width=width, height=height)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        self.root.title(f"Video Annotation App - File: {self.current_file} - Frame: {self.current_frame}")

        # Draw the boxes for the current frame
        self.update_annotation_listbox()

    def step_backward(self, event):
        if self.current_frame is not None and self.current_frame > 0:
            self.current_frame -= 5
            if self.current_frame < 0:
                self.current_frame = 0
            self.seek_frame()

    def step_forward(self, event):
        if self.current_frame is not None:
            self.current_frame += 5
            self.seek_frame()

    def set_class(self, class_id):
        self.current_class = class_id
        # Enable drawing of bounding box with the selected class color
        self.enable_box_drawing()

    def save_annotations(self):
        self.save_results()

    def draw_annotation_box(self, x, y, w, h):
        # Draw a bounding box on the canvas
        pass

    def update_annotation_listbox(self):

        self.annotation_listbox.delete(0, tk.END)  # Clear the listbox

        current_frame_annotations = self.annotated_boxes.get(self.current_frame, [])

        for annotation in current_frame_annotations:
            self.annotation_listbox.insert(tk.END, annotation)

        self.draw_boxes_on_canvas()

    def start_drawing_box(self, event):
        # Capture the starting coordinates for the bounding box
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def draw_box(self, event):
        # Draw the bounding box while dragging the mouse
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        if self.current_box:
            self.canvas.delete(self.current_box)  # Remove the previous box
        # self.canvas.coords(self.box, self.start_x, self.start_y, x, y)
        self.current_box = self.canvas.create_rectangle(self.start_x, self.start_y, x, y,
                                                        outline="gray")  # self.get_color(self.current_class))

    def draw_boxes_on_canvas(self):
        # Clear the canvas before drawing new boxes
        self.canvas.delete("boxes")

        annotations = self.annotated_boxes.get(self.current_frame, [])
        for annotation in annotations:
            class_id, x, y, w, h = map(float, annotation.split())
            # Convert YOLO format to box coordinates
            x1 = (x - w / 2) * self.canvas.winfo_width()
            y1 = (y - h / 2) * self.canvas.winfo_height()
            x2 = (x + w / 2) * self.canvas.winfo_width()
            y2 = (y + h / 2) * self.canvas.winfo_height()

            # Draw the boxes on the canvas
            color = ["red", "green", "cyan", "magenta"][int(class_id)]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, tags="boxes")

    def end_drawing_box(self, event):
        # Capture the ending coordinates for the bounding box and create the annotation
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        # Convert coordinates to yolo format
        x = (self.start_x + end_x) / 2 / self.canvas.winfo_width()
        y = (self.start_y + end_y) / 2 / self.canvas.winfo_height()
        w = abs(end_x - self.start_x) / self.canvas.winfo_width()
        h = abs(end_y - self.start_y) / self.canvas.winfo_height()

        # Truncate coordinates to 3 decimal places
        x, y, w, h = round(x, 3), round(y, 3), round(w, 3), round(h, 3)

        # Store the annotation in the list
        annotation = f"{self.current_class} {x} {y} {w} {h}"
        # self.annotation_boxes.append(annotation)
        self.annotated_boxes[self.current_frame].append(annotation)

        # Draw the bounding box on the canvas
        self.draw_boxes_on_canvas()
        # self.canvas.create_rectangle(self.start_x, self.start_y, end_x, end_y, outline=class_colors[self.current_class])

        # Update the annotation listbox
        self.update_annotation_listbox()

    def enable_box_drawing(self):
        # Bind mouse events to draw bounding boxes
        self.canvas.bind("<Button-1>", self.start_drawing_box)
        self.canvas.bind("<B1-Motion>", self.draw_box)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing_box)

    def save_results(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for frame_number, annotations in self.annotated_boxes.items():
            if annotations:
                frame_name = f"{os.path.splitext(os.path.basename(self.current_file))[0]}_fr{frame_number}"
                frame_path = os.path.join(self.output_folder, f"{frame_name}.png")
                txt_path = os.path.join(self.output_folder, f"{frame_name}.txt")

                # Get the corresponding frame from the video

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = self.cap.read()

                if ret:
                    cv2.imwrite(frame_path, frame)

                    with open(txt_path, 'w') as file:
                        for annotation in annotations:
                            file.write(annotation + '\n')

                    print(f"Saved frame {frame_name}.png with annotations to {self.output_folder}")
                else:
                    print(f"Error saving frame {frame_name}.png")

    def save_state_and_exit(self):
        state_file_path = "state.txt"
        state_file = open(state_file_path, "w")

        if self.current_file:
            state_file.write(f"VideoPath:{self.current_file}\n")
            state_file.write(f"FramePos:{self.current_frame}\n")

        with open(f"{self.current_file}.json", "w") as file:
            file.write(json.dumps(self.annotated_boxes))

        state_file.close()
        self.root.destroy()


    def run(self):
        state_file_path = "state.txt"
        if os.path.isfile(state_file_path):
            state_file = open(state_file_path, "r")
            lines = state_file.readlines()
            state_file.close()

            state = {}
            for line in lines:
                key, value = line.strip().split(":")
                state[key] = value

            if 'VideoPath' in state:
                # self.current_file = state['VideoPath']
                self.open_stream(state['VideoPath'])
                frame_pos = int(float(state['FramePos']))  # Convert to float and then to int
                self.current_frame = frame_pos
                self.seek_frame()

                if os.path.isfile(f"{self.current_file}.json"):
                    with open(f"{self.current_file}.json", "r") as file:
                        self.annotated_boxes = json.loads(file.read(), object_hook=jsonKeys2int)
                print(self.annotated_boxes)

        self.root.mainloop()


def save_annotation_and_frame(video_file, frame_num, annotation_box):
    # Implement this function to save annotation and frame
    pass


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotationApp(root)
    app.run()
