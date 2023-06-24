import json
import os
import random

import cv2
import numpy as np
from PySide6.QtWidgets import QPushButton, QWidget, QFileDialog, QVBoxLayout, QLineEdit, QLabel

from segment_anything_ui.annotator import MasksAnnotation

class FilesHolder:
    def __init__(self):
        self.files = []
        self.position = 0

    def add_files(self, files):
        self.files.extend(files)

    def get_next(self):
        self.position += 1
        if self.position >= len(self.files) - 1:
            self.position = 0
        return self.files[self.position]

    def get_previous(self):
        self.position -= 1

        if self.position < 0:
            self.position = len(self.files) - 1
        return self.files[self.position]


class SettingsLayout(QWidget):
    MASK_EXTENSION = "_mask.png"
    LABELS_EXTENSION = "_labels.json"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.actual_file: str = ""
        self.layout = QVBoxLayout(self)
        self.open_files = QPushButton("Open Files")
        self.open_files.clicked.connect(self.on_open_files)
        self.next_file = QPushButton("Next File")
        self.previous_file = QPushButton("Previous file")
        self.previous_file.setShortcut("D")
        self.save_mask = QPushButton("Save Mask")
        self.save_mask.clicked.connect(self.on_save_mask)
        self.save_mask.setShortcut("Ctrl+S")
        self.next_file.clicked.connect(self.on_next_file)
        self.next_file.setShortcut("F")
        self.previous_file.clicked.connect(self.on_previous_file)
        self.checkpoint_path_label = QLabel(self, text="Checkpoint Path")
        self.checkpoint_path = QLineEdit(self, text=self.parent().config.default_weights)
        self.precompute_button = QPushButton("Precompute all embeddings")
        self.precompute_button.clicked.connect(self.on_precompute)
        self.show_image = QPushButton("Show Image")
        self.show_visualization = QPushButton("Show Visualization")
        self.show_image.clicked.connect(self.on_show_image)
        self.show_visualization.clicked.connect(self.on_show_visualization)
        self.layout.addWidget(self.open_files)
        self.layout.addWidget(self.next_file)
        self.layout.addWidget(self.save_mask)
        self.layout.addWidget(self.checkpoint_path_label)
        self.layout.addWidget(self.checkpoint_path)
        self.checkpoint_path.returnPressed.connect(self.on_checkpoint_path_changed)
        self.checkpoint_path.editingFinished.connect(self.on_checkpoint_path_changed)
        self.layout.addWidget(self.precompute_button)
        self.layout.addWidget(self.show_image)
        self.layout.addWidget(self.show_visualization)
        self.files = FilesHolder()

    def on_next_file(self):
        file = self.files.get_next()
        self._load_image(file)

    def on_previous_file(self):
        file = self.files.get_previous()
        self._load_image(file)

    def _load_image(self, file: str):
        mask = file.split(".")[0] + self.MASK_EXTENSION
        labels = file.split(".")[0] + self.LABELS_EXTENSION
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        self.actual_file = file
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.dtype in [np.float32, np.float64, np.uint16]:
            image = (image / np.amax(image) * 255).astype("uint8")
        image = cv2.resize(image,
                           (self.parent().config.window_size, self.parent().config.window_size))  # TODO: Remove this
        self.parent().annotator.clear()
        self.parent().image_label.clear()
        if os.path.exists(mask) and os.path.exists(labels):
            self._load_annotation(mask, labels)
        self.parent().set_image(image)
        self.parent().update(image)

    def _load_annotation(self, mask, labels):
        mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
        with open(labels, "r") as fp:
            labels: dict[str, str] = json.load(fp)
        masks = []
        new_labels = []
        for str_index, class_ in labels.items():
            single_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            single_mask[mask == int(str_index)] = 1
            masks.append(single_mask)
            new_labels.append(class_)
        self.parent().annotator.masks = MasksAnnotation.from_masks(masks, new_labels)

    def on_show_image(self):
        pass

    def on_show_visualization(self):
        pass

    def on_precompute(self):
        pass

    def on_save_mask(self):
        path = os.path.split(self.actual_file)[0]
        basename = os.path.splitext(os.path.basename(self.actual_file))[0]
        mask_path = os.path.join(path, basename + self.MASK_EXTENSION)
        labels_path = os.path.join(path, basename + self.LABELS_EXTENSION)
        masks = self.parent().get_mask()
        labels = self.parent().get_labels()
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=4)
        cv2.imwrite(mask_path, masks)

    def on_checkpoint_path_changed(self):
        self.parent().sam = self.parent().init_sam()

    def on_open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        random.shuffle(files)
        self.files.add_files(files)
        self.on_next_file()
