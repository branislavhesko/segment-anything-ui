import dataclasses
import os

from PySide6.QtCore import Qt


@dataclasses.dataclass(frozen=True)
class Keymap:
    key: Qt.Key | str
    name: str


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
    default_weights: str = "sam_vit_h_4b8939.pth"
    label_file: str = "labels.json"
    window_size: tuple[int, int] | int = (1920, 1080)
    key_mapping: KeyBindings = dataclasses.field(default_factory=KeyBindings)

    def __post_init__(self):
        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, self.window_size)

    def get_sam_model_name(self):
        if "l2" in self.default_weights:
            return "l2"
        if "vit_b" in self.default_weights:
            return "vit_b"
        if "vit_h" in self.default_weights:
            return "vit_h"
        if "vit_l" in self.default_weights:
            return "vit_l"
        raise ValueError("Unknown model name")
