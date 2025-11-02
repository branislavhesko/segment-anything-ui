import abc

import numpy as np


class CustomModel(abc.ABC):
    
    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        pass
    

class CustomSegmentationModel(CustomModel):
    pass


class CustomDetectionModel(CustomModel):
    pass
