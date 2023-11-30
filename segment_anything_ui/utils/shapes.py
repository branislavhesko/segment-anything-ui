import dataclasses

import cv2
import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QPolygon


@dataclasses.dataclass
class BoundingBox:
    xstart: float
    ystart: float
    xend: float = -1.
    yend: float = -1.

    def to_numpy(self):
        return np.array([self.xstart, self.ystart, self.xend, self.yend])

    def scale(self, sx, sy):
        return BoundingBox(
            xstart=self.xstart * sx,
            ystart=self.ystart * sy,
            xend=self.xend * sx,
            yend=self.yend * sy
        )


@dataclasses.dataclass
class Polygon:
    points: list = dataclasses.field(default_factory=list)

    def to_numpy(self):
        return np.array(self.points).reshape(-1, 2)

    def to_mask(self, num_rows, num_cols):
        mask = np.zeros((num_rows, num_cols))
        mask = cv2.fillPoly(mask, pts=[self.to_numpy(), ], color=255)
        return mask

    def is_plotable(self):
        return len(self.points) > 3

    def to_qpolygon(self):
        return QPolygon([
            QPoint(x, y) for x, y in self.points
        ])
