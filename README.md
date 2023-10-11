# data_review
Visual Data reviewing tool

## Installation
install simply by cloning the repo. create a virtualenv and install dependencies by pip install -r requirements.txt

## Usage
### Setup path
In the source code of the review tool (line 22: self.load_image_paths(folder='new_batches/world_0_drones_1') you can change the name of the data folder. The path starts from the scripts directory.
This folder must exist and the subfolder structure should look like this:
<pre>
new_batches/world_0_drones_1
├── backgrounds
├── out_bbox
├── out_depth
├── out_rgb
├── out_rgb_bbox
└── out_segm
</pre>
backgrounds has the same substructure:
<pre>
new_batches/world_0_drones_1/backgrounds/
├── out_bbox
├── out_depth
├── out_rgb
└── out_segm
</pre>
![image](https://github.com/saab/data_review/assets/6775811/b9825be4-b0f0-463f-a889-dd5c40c54173)


### Reviewing process
When reviewing, look for objects present in real image that does not have a correct bounding box (missing or ill-shaped). A good source of help is to use the semantic view which colours the pixels of our classes of interest.
Find the discrepancies between semantic view and bbox view and judge if you think the object is visible enough to be annotated. If you find a problem with the image, stage it for removal with the associated button.

### Shortcut keys
For convenience, you kan use Right/Left arrow keys to navigate between images. Use Spacebar to stage for removal.

### Removing samples
The images you stage for removal will have a boolean True value in the flagged for deletion field. Whenever you feel like "saving your results" click the update script button. 
This will sift through all images in the listed directory and add a one-line find command to the shell script delete_script.sh of the following form:
<pre>
find . -name \169686183078.* -type f -delete
</pre>
This command will find all images and associated files related to a specific scene.
You may review this file at any time and execute it manually when you feel it is correct.

Note: EVERY TIME you click the update script button, this file will be removed and written from scratch. If you split your work in several passes we suggest using a new script file name for each review session to not interfere with previous revie work.
