import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QPainter, QPen

from segment_anything_ui.image_pixmap import ImagePixmap


class DrawLabel(QtWidgets.QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.positive_points = []
        self.negative_points = []

    def paintEvent(self, paint_event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap())
        pen_positive = self._get_pen(QtCore.Qt.green, 5)
        pen_negative = self._get_pen(QtCore.Qt.red, 5)
        painter.setPen(pen_positive)
        painter.setRenderHint(QPainter.Antialiasing, False)
        for pos in self.positive_points:
            painter.drawPoint(pos)
        painter.setPen(pen_negative)
        painter.setRenderHint(QPainter.Antialiasing, False)
        for pos in self.negative_points:
            painter.drawPoint(pos)
        # self.update()

    def _get_pen(self, color=QtCore.Qt.red, width=5):
        pen = QPen()
        pen.setWidth(width)
        pen.setColor(color)
        return pen

    def mouseReleaseEvent(self, cursor_event):
        if cursor_event.button() == QtCore.Qt.LeftButton:
            self.positive_points.append(cursor_event.pos())
        elif cursor_event.button() == QtCore.Qt.RightButton:
            self.negative_points.append(cursor_event.pos())
        # self.chosen_points.append(self.mapFromGlobal(QtGui.QCursor.pos()))
        self.parent().annotator.make_prediction()
        self.parent().annotator.visualize_last_mask()
        self.update()

    def get_points(self):
        positive_points = np.array(self.positive_points).reshape(-1, 2)
        negative_points = np.array(self.negative_points).reshape(-1, 2)
        labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        return {
            "points": np.concatenate([positive_points, negative_points], axis=0),
            "labels": labels
        }

    def clear_points(self):
        self.positive_points = []
        self.negative_points = []
        self.update()
