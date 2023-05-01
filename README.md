# Segment Anything UI
Simple UI for the model: [Segment anything](https://github.com/facebookresearch/segment-anything) from Facebook.


Segment anything UI for annotations
![GUI](./assets/parrots.png)



# Usage

 1. Install segment-anything python package from Github: [Segment anything](https://github.com/facebookresearch/segment-anything).
 2. Download checkpoint [Checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and put it into workspace folder.
 3. Fill default_path in config.py.
 4. Install requirements.txt. ```pip install -r requirements.txt```.
 5. ```export PYTHONPATH=$PYTHONPATH:.```.
 6. ```python segment_anything_ui/main_window.py```.

Currently, for saving a simple format is used: mask is saved as .png file, when masks are represented by values: 1 ... n and corresponding labels are saved as jsons. In json, labels are a map with mapping: MASK_ID: LABEL. MASK_ID is the id of the stored mask and LABEL is one of "labels.json" files.


# TODO:

 - [ ] - FIX: mouse picker for small objects is not precise.
 - [ ] - Region merging.
 - [ ] - Manual annotation + brush + deleting options.
 - [ ] - Shortcut description.
 - [x] - Saving and loading of masks.
 - [x] - Class support for assigning classes to objects.
 - [x] - Add object borders.
