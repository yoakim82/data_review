import os
import tkinter as tk
from PIL import Image, ImageTk
import colorpalette

class ImageRemovalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Removal Tool")

        self.image_paths_rgb_bbox = []
        self.image_paths_segm = []
        self.image_paths_depth = []

        self.current_index = 0
        self.script_file_name = "delete_script.sh"  # Default script file name

        self.load_image_paths()
        self.load_images()

        self.create_gui()

    def load_image_paths(self):
        for root, _, _ in os.walk('data'):
            if 'out_rgb_bbox' in root:
                files = os.listdir(root)
                files = [os.path.join(root, f) for f in files]
                files.sort(key=lambda x: os.path.getmtime(x))
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # Add more image extensions if needed
                        self.image_paths_rgb_bbox.append(file)
                        base = os.path.basename(file)
                        dirname = os.path.split(root)[0]
                        self.image_paths_segm.append(os.path.join(dirname, 'out_segm', base))
                        self.image_paths_depth.append(os.path.join(dirname, 'out_depth', base))

    def load_images(self):
        self.images_rgb_bbox = [Image.open(file) for file in self.image_paths_rgb_bbox]
        self.images_segm = [Image.open(file) for file in self.image_paths_segm]
        self.images_depth = [Image.open(file) for file in self.image_paths_depth]

    def create_gui(self):
        self.upper_left_label = tk.Label(root)
        self.lower_left_label = tk.Label(root)
        self.bottom_right_label = tk.Label(root)

        self.upper_left_label.grid(row=0, column=0)
        self.lower_left_label.grid(row=1, column=0)
        self.bottom_right_label.grid(row=1, column=1)

        self.next_button = tk.Button(root, text="Next", command=self.next_image)
        self.prev_button = tk.Button(root, text="Prev", command=self.prev_image)

        self.remove_button = tk.Button(root, text="Remove", command=self.remove_image)
        self.script_label = tk.Label(root, text="Script File Name:")
        self.script_entry = tk.Entry(root)
        self.script_entry.insert(0, self.script_file_name)
        self.update_script_button = tk.Button(root, text="Update Script", command=self.update_script)

        self.next_button.grid(row=0, column=2, rowspan=1, columnspan=1)
        self.prev_button.grid(row=0, column=1, rowspan=1, columnspan=1)
        self.remove_button.grid(row=0, column=3, rowspan=1, columnspan=1)
        self.script_label.grid(row=2, column=0, rowspan=1, columnspan=1)
        self.script_entry.grid(row=2, column=1, rowspan=1, columnspan=1)
        self.update_script_button.grid(row=2, column=2, rowspan=1, columnspan=1)

        self.load_quadrant_images()

    def load_quadrant_images(self):
        scale =3
        self.root.title(f"image {self.current_index}: {self.image_paths_rgb_bbox[self.current_index]}")
        img_format = (int(1920 / scale), int(1080 / scale))
        if self.current_index < len(self.image_paths_rgb_bbox):
            image_rgb_bbox = self.images_rgb_bbox[self.current_index].resize(img_format)
            photo_rgb_bbox = ImageTk.PhotoImage(image_rgb_bbox)
            self.upper_left_label.config(image=photo_rgb_bbox)
            self.upper_left_label.image = photo_rgb_bbox

            image_segm = self.images_segm[self.current_index].resize(img_format, resample=Image.Resampling.NEAREST)
            transform_segm = colorpalette.convert_image_with_palette(image_segm)
            photo_segm = ImageTk.PhotoImage(transform_segm)
            self.lower_left_label.config(image=photo_segm)
            self.lower_left_label.image = photo_segm

            image_depth = self.images_depth[self.current_index].resize(img_format)
            photo_depth = ImageTk.PhotoImage(image_depth)
            self.bottom_right_label.config(image=photo_depth)
            self.bottom_right_label.image = photo_depth

    def next_image(self):
        self.current_index += 1
        if self.current_index < len(self.image_paths_rgb_bbox):
            self.load_quadrant_images()
        else:
            self.upper_left_label.config(text="No more images")
            self.lower_left_label.config(text="")
            self.bottom_right_label.config(text="")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_quadrant_images()

    def remove_image(self):
        if self.current_index < len(self.image_paths_rgb_bbox):
            file_name = os.path.basename(self.image_paths_rgb_bbox[self.current_index])
            file_prefix = file_name.split('_')[0]
            self.append_to_script(f"rm {file_prefix}*")
            del self.image_paths_rgb_bbox[self.current_index]
            del self.images_rgb_bbox[self.current_index]
            del self.images_segm[self.current_index]
            del self.images_depth[self.current_index]
            self.load_quadrant_images()

    def append_to_script(self, command):
        with open(self.script_file_name, "a") as script_file:
            script_file.write(command + "\n")

    def update_script(self):
        self.script_file_name = self.script_entry.get()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRemovalApp(root)
    root.mainloop()
