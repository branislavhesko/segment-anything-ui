import dataclasses
import os

from PySide6.QtCore import Qt


@dataclasses.dataclass(frozen=True)
class Keymap:
    key: Qt.Key
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
    SAVE_MASK: Keymap = Keymap(Qt.Key.Key_S, "Ctrl+S")
    PRECOMPUTE: Keymap = Keymap(Qt.Key.Key_P, "P")


@dataclasses.dataclass
class Config:
    default_weights: str = "sam_vit_b_01ec64.pth"
    label_file: str = "labels.json"
    window_size: tuple[int, int] | int = (1600, 900)
    key_mapping: KeyBindings = dataclasses.field(default_factory=KeyBindings)

    def __post_init__(self):
        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, self.window_size)

    def get_model_name(self):
        return "_".join(os.path.basename(self.default_weights).split("_")[1:3])
