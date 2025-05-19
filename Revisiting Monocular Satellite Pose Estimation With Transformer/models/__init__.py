from .detr_speed import build
from .detr_speed import PostProcess


def build_model(args):
    return build(args)
