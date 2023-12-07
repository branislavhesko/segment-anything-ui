import glob
import os

import cv2
import numpy as np
import torch
import rich
from PIL import Image
import safetensors
from segment_anything import sam_model_registry

from segment_anything_ui.modeling.storable_sam import StorableSam
from segment_anything_ui.config import Config

config = Config()
sam = sam_model_registry[config.get_sam_model_name()](checkpoint=config.default_weights)
allowed_extensions = [".jpg", ".png", ".tif", ".tiff"]


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        allowed_extensions = [".jpg", ".png"]
        if any(filename.endswith(ext) for ext in allowed_extensions):
            img = Image.open(os.path.join(folder_path, filename))
    return images
