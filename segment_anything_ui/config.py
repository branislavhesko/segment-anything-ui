import dataclasses


@dataclasses.dataclass
class Config:
    window_size: int = 1024
    default_weights = "sam_vit_b_01ec64.pth"

    def get_model_name(self):
        return "_".join(self.default_weights.split("_")[1:3])
