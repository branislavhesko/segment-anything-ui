import os
from PySide6.QtWidgets import QMessageBox

try:
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model, EfficientViTSam
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor, EfficientViTSamAutomaticMaskGenerator
    IS_EFFICIENT_VIT_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    import logging
    logging.warning("Efficient is not available, please install the package from https://github.com/mit-han-lab/efficientvit/tree/master .")
    IS_EFFICIENT_VIT_AVAILABLE = False
    
try:
    from segment_anything_hq import sam_model_registry as sam_hq_model_registry
    from segment_anything_hq import SamPredictor as SamPredictorHQ
    from segment_anything_hq import automatic_mask_generator as automatic_mask_generator_hq
    from segment_anything_hq.build_sam import Sam as SamHQ
    IS_SAM_HQ_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    import logging
    logging.warning("Segment Anything HQ is not available, please install the package from http://github.com/SysCV/sam-hq .")
    IS_SAM_HQ_AVAILABLE = False
    

try:
    from sam2.build_sam import build_sam2
    from sam2.modeling.sam2_base import SAM2Base
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    IS_SAM2_AVAILABLE = True
    
    _SAM2_MODEL_REGISTRY = {
        "sam2.1_hiera_t": os.path.join(os.path.dirname(__file__), "sam2_configs", "sam2.1_hiera_t.yaml"),
        "sam2.1_hiera_l": os.path.join(os.path.dirname(__file__), "sam2_configs", "sam2.1_hiera_l.yaml"),
        "sam2.1_hiera_b": os.path.join(os.path.dirname(__file__), "sam2_configs", "sam2.1_hiera_b+.yaml"),
        "sam2.1_hiera_s": os.path.join(os.path.dirname(__file__), "sam2_configs", "sam2.1_hiera_s.yaml"),
    }
    
except (ModuleNotFoundError, ImportError) as e:
    import logging
    logging.warning("SAM2 is not available, please install the package from https://github.com/SysCV/sam2 .")
    IS_SAM2_AVAILABLE = False
    _SAM2_MODEL_REGISTRY = {}

from segment_anything import sam_model_registry
from segment_anything import SamPredictor, automatic_mask_generator
from segment_anything.build_sam import Sam


def build_model(model_name: str, checkpoint_path: str, device: str):
    match model_name:
        case "xl0" | "xl1":
            if not IS_EFFICIENT_VIT_AVAILABLE:
                raise ValueError("EfficientViTSam is not available, please install the package from https://github.com/mit-han-lab/efficientvit/tree/master .")
            efficientvit_sam = create_efficientvit_sam_model(
                name=model_name, weight_url=checkpoint_path,
            )
            return efficientvit_sam.to(device).eval()
        
        case "vit_b" | "vit_l" | "vit_h":
            sam = sam_model_registry[model_name](
                checkpoint=checkpoint_path)
            sam.eval()
            return sam.to(device)
        
        case "hq_vit_b" | "hq_vit_l" | "hq_vit_h":
            if not IS_SAM_HQ_AVAILABLE:
                QMessageBox.critical(None, "Segment Anything HQ is not available", "Please install the package from http://github.com/SysCV/sam-hq .")
                raise ValueError("Segment Anything HQ is not available, please install the package from http://github.com/SysCV/sam-hq .")
            sam = sam_hq_model_registry[model_name](
                checkpoint=checkpoint_path)
            sam.eval()
            return sam.to(device)
        case "sam2.1_hiera_t" | "sam2.1_hiera_l" | "sam2.1_hiera_b" | "sam2.1_hiera_s":
            if not IS_SAM2_AVAILABLE:
                QMessageBox.critical(None, "SAM2 is not available", "Please install the package from https://github.com/facebookresearch/sam2 .")
                raise ValueError("SAM2 is not available, please install the package from https://github.com/facebookresearch/sam2 .")
            sam = build_sam2(_SAM2_MODEL_REGISTRY[model_name], checkpoint_path, device=device)
            sam.eval()
            return sam
        case _:
            raise ValueError(f"Model {model_name} not supported")
        
        
def get_predictor(sam):
    if isinstance(sam, Sam):
        return SamPredictor(sam)    
    elif IS_EFFICIENT_VIT_AVAILABLE and isinstance(sam, EfficientViTSam):
        return EfficientViTSamPredictor(sam)
    
    elif IS_SAM_HQ_AVAILABLE and isinstance(sam, SamHQ):
        return SamPredictorHQ(sam)
    
    elif IS_SAM2_AVAILABLE and isinstance(sam, SAM2Base):
        return SAM2ImagePredictor(sam)
    else:
        raise ValueError("Model is not an EfficientViTSam or Sam")    


def get_mask_generator(sam, **kwargs):
    if isinstance(sam, Sam):
        return automatic_mask_generator.SamAutomaticMaskGenerator(model=sam, **kwargs)
    
    elif IS_SAM_HQ_AVAILABLE and isinstance(sam, SamHQ):
        return automatic_mask_generator_hq.SamAutomaticMaskGeneratorHQ(model=sam, **kwargs)
    
    elif IS_EFFICIENT_VIT_AVAILABLE and isinstance(sam, EfficientViTSam):
        return EfficientViTSamAutomaticMaskGenerator(model=sam, **kwargs)
    
    elif IS_SAM2_AVAILABLE and isinstance(sam, SAM2Base):
        return SAM2AutomaticMaskGenerator(model=sam)
    
    else:
        raise ValueError("Model is not an EfficientViTSam or Sam")