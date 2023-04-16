import copy
import dataclasses
from enum import Enum

import numpy as np
import PySide6
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QPainter, QPen

from segment_anything_ui.image_pixmap import ImagePixmap


class PaintType(Enum):
    POINT = 0
    BOX = 1
    MASK = 2


@dataclasses.dataclass
class BoundingBox:
    xstart: float
    ystart: float
    xend: float = -1.
    yend: float = -1.

    def to_numpy(self):
        return np.array([self.xstart, self.ystart, self.xend, self.yend])


class DrawLabel(QtWidgets.QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.positive_points = []
        self.negative_points = []
        self.bounding_box = None
        self.partial_box = BoundingBox(0, 0)
        self._paint_type = PaintType.POINT

    def paintEvent(self, paint_event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap())
        pen_positive = self._get_pen(QtCore.Qt.green, 5)
        pen_negative = self._get_pen(QtCore.Qt.red, 5)
        pen_partial = self._get_pen(QtCore.Qt.yellow, 1)
        pen_box = self._get_pen(QtCore.Qt.green, 1)
        painter.setRenderHint(QPainter.Antialiasing, False)

        painter.setPen(pen_box)

        if self.bounding_box is not None:
            painter.drawRect(
                self.bounding_box.xstart,
                self.bounding_box.ystart,
                self.bounding_box.xend - self.bounding_box.xstart,
                self.bounding_box.yend - self.bounding_box.ystart
            )

        painter.setPen(pen_partial)
        painter.drawRect(self.partial_box.xstart, self.partial_box.ystart,
                         self.partial_box.xend - self.partial_box.xstart,
                         self.partial_box.yend - self.partial_box.ystart)

        painter.setPen(pen_positive)
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

    def change_paint_type(self, paint_type: PaintType):
        print(f"Changing paint type to {paint_type}")
        self._paint_type = paint_type

    def mouseMoveEvent(self, ev: PySide6.QtGui.QMouseEvent) -> None:
        if self._paint_type == PaintType.BOX:
            self.partial_box = copy.deepcopy(self.bounding_box)
            self.partial_box.xend = ev.pos().x()
            self.partial_box.yend = ev.pos().y()
            self.update()

    def mouseReleaseEvent(self, cursor_event):
        if self._paint_type == PaintType.POINT:
            if cursor_event.button() == QtCore.Qt.LeftButton:
                self.positive_points.append(cursor_event.pos())
                print(self.size())
            elif cursor_event.button() == QtCore.Qt.RightButton:
                self.negative_points.append(cursor_event.pos())
            # self.chosen_points.append(self.mapFromGlobal(QtGui.QCursor.pos()))
        elif self._paint_type == PaintType.BOX:
            if cursor_event.button() == QtCore.Qt.LeftButton:
                self.bounding_box.xend = cursor_event.pos().x()
                self.bounding_box.yend = cursor_event.pos().y()
                self.partial_box = BoundingBox(0, 0, 0, 0)
        self.parent().annotator.make_prediction(self.get_annotations())
        self.parent().annotator.visualize_last_mask()
        self.update()

    def mousePressEvent(self, ev: PySide6.QtGui.QMouseEvent) -> None:
        if self._paint_type == PaintType.BOX and ev.button() == QtCore.Qt.LeftButton:
            self.bounding_box = BoundingBox(xstart=ev.pos().x(), ystart=ev.pos().y())

    def get_annotations(self):
        positive_points = [(p.x(), p.y()) for p in self.positive_points]
        positive_points = np.array(positive_points).reshape(-1, 2)
        negative_points = [(p.x(), p.y()) for p in self.negative_points]
        negative_points = np.array(negative_points).reshape(-1, 2)
        labels = np.array([1, ] * len(positive_points) + [0, ] * len(negative_points))
        print(f"Positive points: {positive_points}")
        print(f"Negative points: {negative_points}")
        print(f"Labels: {labels}")
        return {
            "points": np.concatenate([positive_points, negative_points], axis=0),
            "labels": labels,
            "bounding_boxes": self.bounding_box.to_numpy() if self.bounding_box else None
        }

    def clear(self):
        self.positive_points = []
        self.negative_points = []
        self.bounding_box = None
        self.update()
