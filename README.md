# Segment Anything UI
Simple UI for the model: [Segment anything](https://github.com/facebookresearch/segment-anything) from Facebook.


Segment anything UI for annotations
![GUI](./assets/example.png)



# Usage

 1. Install segment-anything python package from Github: [Segment anything](https://github.com/facebookresearch/segment-anything). Usually it is enough to run: ```pip install git+https://github.com/facebookresearch/segment-anything.git```.
 2. Download checkpoint [Checkpoint_Huge](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) or [Checkpoint_Large](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) or [Checkpoint_Base](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and put it into workspace folder.
 3. Fill default_path in ```segment_anything_ui/config.py```.
 4. Install requirements.txt. ```pip install -r requirements.txt```.
 5. ```export PYTHONPATH=$PYTHONPATH:.```.
 6. ```python segment_anything_ui/main_window.py```.

Currently, for saving a simple format is used: mask is saved as .png file, when masks are represented by values: 1 ... n and corresponding labels are saved as jsons. In json, labels are a map with mapping: MASK_ID: LABEL. MASK_ID is the id of the stored mask and LABEL is one of "labels.json" files.

# Buttons

| **Button** | **Description** | **Shortcut** |
| --- | --- | --- |
| Add Points | Mouse click will add positive (left) or negative (right) point. | W |
| Add Box | Mouse click will add a box. | Q |
| Annotate All | Runs regular grid annotation with parameters from the form | Enter |
| Pick Mask | Pick mask when clicking on it. Cycling through masks if pixel belongs to multiple masks. | X |
| Merge Masks | WIP: Merge masks when clicking on them. Cycling through masks if pixel belongs to multiple masks. | Z |
| Move Current Mask to Front | Use current mask as background (less important) | None |
| Cancel Annotation | Cancel current annotation | C |
| Save Annotation | Save current annotation | S |
| Manual Polygon | Draw polygon with mouse | R |
| Partial Mask | Allows to split a single object into multiple prompts. When pressed, partial mask is stored and summed with the following promped masks. | D |
| ---- | ---- | ---- |
| Open Files | Load files from the folder | None |
| Next File | Load next image | F |
| Previous File | Load previous image | G |
| ---- | ---- | ---- |
| Precompute All Embeddings | Currently not implemented | None |
| Show Image | Currently not implemented | None |
| Show Visualization | Currently not implemented | None |

# TODO:

 - [x] - FIX: mouse picker for small objects is not precise.
 - [ ] - Region merging.
 - [x] - Manual annotation.
 - [x] - Saving and loading of masks.
 - [x] - Class support for assigning classes to objects.
 - [x] - Add object borders.
 - [x] - Fix mask size and QLabel size for precise mouse clicks.
 - [ ] - Draft mask when no points are visible.
 - [x] - Box zoom support.