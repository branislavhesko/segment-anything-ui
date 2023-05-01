import os
from typing import Any
import torch


class Saver:

    def __init__(self, path: str) -> None:
        self.path = path

    def __call__(self, basename, mask_annotation) -> Any:
        save_path = os.path.join(self.path, basename)
        # TODO: finish this