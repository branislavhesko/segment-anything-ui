import cv2
from PySide6.QtWidgets import QPushButton, QWidget, QFileDialog, QVBoxLayout, QLineEdit, QLabel


class FilesHolder:
    def __init__(self):
        self.files = []
        self.position = 0

    def add_files(self, files):
        self.files.extend(files)

    def get_next(self):
        if self.position >= len(self.files):
            self.position = 0
        self.position += 1
        return self.files[self.position - 1]


class SettingsLayout(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.open_files = QPushButton("Open Files")
        self.open_files.clicked.connect(self.on_open_files)
        self.next_file = QPushButton("Next File")
        self.save_mask = QPushButton("Save Mask")
        self.save_mask.clicked.connect(self.on_save_mask)
        self.next_file.clicked.connect(self.on_next_file)
        self.checkpoint_path_label = QLabel(self, text="Checkpoint Path")
        self.checkpoint_path = QLineEdit(self, text="sam_vit_b_01ec64.pth")
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
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))  # TODO: Remove this
        self.parent().set_image(image)

    def on_show_image(self):
        pass

    def on_show_visualization(self):
        pass

    def on_precompute(self):
        pass

    def on_save_mask(self):
        mask = self.parent().get_mask()
        cv2.imwrite("mask.png", mask.astype("uint8"))

    def on_checkpoint_path_changed(self):
        self.parent().sam = self.parent().init_sam()

    def on_open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Image Files (*.png *.jpg *.bmp)")
        self.files.add_files(files)
        self.on_next_file()