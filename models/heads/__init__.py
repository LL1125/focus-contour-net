"""Detection and contour heads."""

from models.heads.detect_head_e2e import DetectHeadE2E
from models.heads.fourier_contour_head import FourierContourHead

__all__ = ["DetectHeadE2E", "FourierContourHead"]
