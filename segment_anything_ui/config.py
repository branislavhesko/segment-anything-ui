import dataclasses
import os


@dataclasses.dataclass
class Config:
    window_size: int = 1088
    default_weights = "sam_vit_b_01ec64.pth"
    label_file = "labels.json"

    def get_model_name(self):
        return "_".join(os.path.basename(self.default_weights).split("_")[1:3])
