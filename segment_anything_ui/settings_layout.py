import json
import os
import random

import cv2
import numpy as np
from PySide6.QtWidgets import QPushButton, QWidget, QFileDialog, QVBoxLayout, QLineEdit, QLabel, QCheckBox, QMessageBox

from segment_anything_ui.annotator import MasksAnnotation
from segment_anything_ui.config import Config


class FilesHolder:
    def __init__(self):
        self.files = []
        self.position = 0

    def add_files(self, files):
        self.files.extend(files)

    def get_next(self):
        self.position += 1
        if self.position >= len(self.files):
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

    def __init__(self, parent, config: Config) -> None:
        super().__init__(parent)
        self.config = config
        self.actual_file: str = ""
        self.actual_shape = self.config.window_size
        self.layout = QVBoxLayout(self)
        self.open_files = QPushButton("Open Files")
        self.open_files.clicked.connect(self.on_open_files)
        self.next_file = QPushButton(f"Next File [ {config.key_mapping.NEXT_FILE.name} ]")
        self.previous_file = QPushButton(f"Previous file [ {config.key_mapping.PREVIOUS_FILE.name} ]")
        self.previous_file.setShortcut(config.key_mapping.PREVIOUS_FILE.key)
        self.save_mask = QPushButton(f"Save Mask [ {config.key_mapping.SAVE_MASK.name} ]")
        self.save_mask.clicked.connect(self.on_save_mask)
        self.save_mask.setShortcut(config.key_mapping.SAVE_MASK.key)
        self.next_file.clicked.connect(self.on_next_file)
        self.next_file.setShortcut(config.key_mapping.NEXT_FILE.key)
        self.previous_file.clicked.connect(self.on_previous_file)
        self.checkpoint_path_label = QLabel(self, text="Checkpoint Path")
        self.checkpoint_path = QLineEdit(self, text=self.parent().config.default_weights)
        self.precompute_button = QPushButton("Precompute all embeddings")
        self.precompute_button.clicked.connect(self.on_precompute)
        self.show_image = QPushButton("Show Image")
        self.show_visualization = QPushButton("Show Visualization")
        self.show_image.clicked.connect(self.on_show_image)
        self.show_visualization.clicked.connect(self.on_show_visualization)
        self.show_text = QCheckBox("Show Text")
        self.show_text.clicked.connect(self.on_show_text)
        self.tag_text_field = QLineEdit(self)
        self.tag_text_field.setPlaceholderText("Comma separated image tags")
        self.layout.addWidget(self.open_files)
        self.layout.addWidget(self.next_file)
        self.layout.addWidget(self.previous_file)
        self.layout.addWidget(self.save_mask)
        self.layout.addWidget(self.show_text)
        self.layout.addWidget(self.tag_text_field)
        self.layout.addWidget(self.checkpoint_path_label)
        self.layout.addWidget(self.checkpoint_path)
        self.checkpoint_path.returnPressed.connect(self.on_checkpoint_path_changed)
        self.checkpoint_path.editingFinished.connect(self.on_checkpoint_path_changed)
        self.layout.addWidget(self.precompute_button)
        self.layout.addWidget(self.show_image)
        self.layout.addWidget(self.show_visualization)
        self.files = FilesHolder()

    def is_show_text(self):
        return self.show_text.isChecked()

    def on_show_text(self):
        self.parent().update(self.parent().annotator.merge_image_visualization())

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
        self.actual_shape = image.shape[:2][::-1]
        self.actual_file = file
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.dtype in [np.float32, np.float64, np.uint16]:
            image = (image / np.amax(image) * 255).astype("uint8")
        image = cv2.resize(image,
                           (int(self.parent().config.window_size[0]), self.parent().config.window_size[1]))
        self.parent().annotator.clear()
        self.parent().image_label.clear()
        self.parent().set_image(image)
        if os.path.exists(mask) and os.path.exists(labels):
            self._load_annotation(mask, labels)
            self.parent().update(self.parent().annotator.merge_image_visualization())
        else:
            self.parent().update(image)

    def _load_annotation(self, mask, labels):
        mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.config.window_size[0], self.config.window_size[1]),
                          interpolation=cv2.INTER_NEAREST)
        with open(labels, "r") as fp:
            labels: dict[str, str] = json.load(fp)
        masks = []
        new_labels = []
        if "instances" in labels:
            instance_labels = labels["instances"]
        else:
            instance_labels = labels

        if "tags" in labels:
            self.tag_text_field.setText(",".join(labels["tags"]))
        for str_index, class_ in instance_labels.items():
            single_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            single_mask[mask == int(str_index)] = 255
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
        tags = self.tag_text_field.text().split(",")
        tags = [tag.strip() for tag in tags]
        basename = os.path.splitext(os.path.basename(self.actual_file))[0]
        mask_path = os.path.join(path, basename + self.MASK_EXTENSION)
        labels_path = os.path.join(path, basename + self.LABELS_EXTENSION)
        masks = self.parent().get_mask()
        labels = {"instances": self.parent().get_labels(), "tags": tags}
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=4)
        masks = cv2.resize(masks, self.actual_shape, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(mask_path, masks)

    def on_checkpoint_path_changed(self):
        self.parent().sam = self.parent().init_sam()

    def on_open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        random.shuffle(files)
        self.files.add_files(files)
        self.on_next_file()
