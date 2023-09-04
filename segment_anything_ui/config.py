import dataclasses
import os


@dataclasses.dataclass
class Config:
    window_size: tuple[int, int] | int = (1600, 900)
    default_weights = "sam_vit_b_01ec64.pth"
    label_file = "labels.json"

    def __post_init__(self):
        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, self.window_size)

    def get_model_name(self):
        return "_".join(os.path.basename(self.default_weights).split("_")[1:3])
