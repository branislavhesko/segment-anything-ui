from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QPainter, QPen


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
        self.update()
