import dataclasses
import os
from typing import Literal

from PySide6.QtCore import Qt
import requests
try:
    from tqdm import tqdm
    import wget
except ImportError:
    print("Tqdm and wget not found. Install with pip install tqdm wget")
    tqdm = None
    wget = None


@dataclasses.dataclass(frozen=True)
class Keymap:
    key: Qt.Key | str
    name: str


class ProgressBar:
    def __init__(self):
        self.progress_bar = None

    def __call__(self, current_bytes, total_bytes, width):
        current_mb = round(current_bytes / 1024 ** 2, 1)
        total_mb = round(total_bytes / 1024 ** 2, 1)
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=total_mb, desc="MB")
        delta_mb = current_mb - self.progress_bar.n
        self.progress_bar.update(delta_mb)


@dataclasses.dataclass
class KeyBindings:
    ADD_POINT: Keymap = Keymap(Qt.Key.Key_W, "W")
    ADD_BOX: Keymap = Keymap(Qt.Key.Key_Q, "Q")
    ANNOTATE_ALL: Keymap = Keymap(Qt.Key.Key_Return, "Enter")
    MANUAL_POLYGON: Keymap = Keymap(Qt.Key.Key_R, "R")
    CANCEL_ANNOTATION: Keymap = Keymap(Qt.Key.Key_C, "C")
    SAVE_ANNOTATION: Keymap = Keymap(Qt.Key.Key_S, "S")
    PICK_MASK: Keymap = Keymap(Qt.Key.Key_X, "X")
    MERGE_MASK: Keymap = Keymap(Qt.Key.Key_Z, "Z")
    DELETE_MASK: Keymap = Keymap(Qt.Key.Key_V, "V")
    PARTIAL_ANNOTATION: Keymap = Keymap(Qt.Key.Key_D, "D")

    NEXT_FILE: Keymap = Keymap(Qt.Key.Key_F, "F")
    PREVIOUS_FILE: Keymap = Keymap(Qt.Key.Key_G, "G")
    SAVE_MASK: Keymap = Keymap("Ctrl+S", "Ctrl+S")
    PRECOMPUTE: Keymap = Keymap(Qt.Key.Key_P, "P")
    ZOOM_RECTANGLE: Keymap = Keymap(Qt.Key.Key_E, "E")


@dataclasses.dataclass
class Config:
    default_weights: Literal[
        "sam_vit_b_01ec64.pth", 
        "sam_vit_h_4b8939.pth", 
        "sam_vit_l_0b3195.pth", 
        "xl0.pt", 
        "xl1.pt", 
        "PATH_TO_YOUR_CHECKPOINT"
    ] = "xl0.pt"
    download_weights_if_not_available: bool = True
    label_file: str = "labels.json"
    window_size: tuple[int, int] | int = (1920, 1080)
    key_mapping: KeyBindings = dataclasses.field(default_factory=KeyBindings)
    weights_paths: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "l2": "https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l2.pt",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "xl0": "https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl0.pt",
        "xl1": "https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt",
    })

    def __post_init__(self):
        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, self.window_size)
        if self.download_weights_if_not_available:
            self.download_weights()

    def get_sam_model_name(self):
        if "l2" in self.default_weights:
            return "l2"
        if "vit_b" in self.default_weights:
            return "vit_b"
        if "vit_h" in self.default_weights:
            return "vit_h"
        if "vit_l" in self.default_weights:
            return "vit_l"
        if "xl0" in self.default_weights:
            return "xl0"
        if "xl1" in self.default_weights:
            return "xl1"
        raise ValueError("Unknown model name")

    def download_weights(self):
        if not os.path.exists(self.default_weights):
            try:
                print(f"Downloading weights for model {self.get_sam_model_name()}")
                wget.download(self.weights_paths[self.get_sam_model_name()], self.default_weights, bar=ProgressBar())
                print(f"Downloaded weights to {self.default_weights}")
            except Exception as e:
                print(f"Error downloading weights: {e}. Trying with requests.")
                model_name = self.get_sam_model_name()
                print(f"Downloading weights for model {model_name}")
                file = requests.get(self.weights_paths[model_name])
                with open(self.default_weights, "wb") as f:
                    f.write(file.content)
                print(f"Downloaded weights to {self.default_weights}")
