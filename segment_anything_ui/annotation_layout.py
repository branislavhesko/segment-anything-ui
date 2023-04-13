from PySide6.QtWidgets import QWidget, QLabel, QSpinBox, QDoubleSpinBox, QVBoxLayout, QPushButton


class AnnotationLayoutSettings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.num_points_label = QLabel("Number of Points:")
        self.num_points_input = QSpinBox()
        self.num_points_input.setMinimum(1)
        self.num_points_input.setMaximum(1000)
        self.num_points_input.setValue(10)

        # Create the IOU threshold label and input field
        self.iou_threshold_label = QLabel("IOU Threshold:")
        self.iou_threshold_input = QDoubleSpinBox()
        self.iou_threshold_input.setMinimum(0.0)
        self.iou_threshold_input.setMaximum(1.0)
        self.iou_threshold_input.setValue(0.5)
        self.layout.addWidget(self.num_points_label)
        self.layout.addWidget(self.num_points_input)
        self.layout.addWidget(self.iou_threshold_label)
        self.layout.addWidget(self.iou_threshold_input)


class AnnotationLayout(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.add_point = QPushButton("Add Point")
        self.add_box = QPushButton("Add Box")
        self.annotate_all = QPushButton("Annotate All")
        self.cancel_annotation = QPushButton("Cancel Annotation")
        self.save_annotation = QPushButton("Save Annotation")
        self.annotation_settings = AnnotationLayoutSettings(self)
        self.layout.addWidget(self.add_point)
        self.layout.addWidget(self.add_box)
        self.layout.addWidget(self.annotate_all)
        self.layout.addWidget(self.cancel_annotation)
        self.layout.addWidget(self.save_annotation)
        self.layout.addWidget(self.annotation_settings)
        self.add_point.clicked.connect(self.on_add_point)
        self.add_box.clicked.connect(self.on_add_box)
        self.annotate_all.clicked.connect(self.on_annotate_all)
        self.cancel_annotation.clicked.connect(self.on_cancel_annotation)
        self.save_annotation.clicked.connect(self.on_save_annotation)

    def on_add_point(self):
        pass

    def on_add_box(self):
        pass

    def on_annotate_all(self):
        pass

    def on_cancel_annotation(self):
        self.parent().image_label.clear_points()

    def on_save_annotation(self):
        pass
