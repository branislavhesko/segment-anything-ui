import cv2
from PySide6.QtWidgets import QPushButton, QWidget, QFileDialog, QGridLayout, QLineEdit, QLabel


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
        self.layout = QGridLayout(self)
        self.open_files = QPushButton("Open Files")
        self.open_files.clicked.connect(self.on_open_files)
        self.next_file = QPushButton("Next File")
        self.save_mask = QPushButton("Save Mask")
        self.save_mask.clicked.connect(self.on_save_mask)
        self.next_file.clicked.connect(self.on_next_file)
        self.checkpoint_path_label = QLabel(self, text="Checkpoint Path")
        self.checkpoint_path = QLineEdit(self, text="ADD.pth")
        self.precompute_button = QPushButton("Precompute all embeddings")
        self.precompute_button.clicked.connect(self.on_precompute)
        self.layout.addWidget(self.open_files, 0, 0)
        self.layout.addWidget(self.next_file, 1, 0)
        self.layout.addWidget(self.save_mask, 2, 0)
        self.layout.addWidget(self.checkpoint_path_label, 3, 0)
        self.layout.addWidget(self.checkpoint_path, 4, 0)
        self.files = FilesHolder()

    def on_next_file(self):
        file = self.files.get_next()
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512)) # TODO: Remove this
        self.parent().update(image)

    def on_precompute(self):
        pass

    def on_save_mask(self):
        pass

    def on_open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Image Files (*.png *.jpg *.bmp)")
        self.files.add_files(files)
        self.on_next_file()