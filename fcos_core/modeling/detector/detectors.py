# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_rcnn_pose import GeneralizedRCNNPose


# _DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}

_DETECTION_META_ARCHITECTURES = {
    "GeneralizedRCNN": GeneralizedRCNN,
    "GeneralizedRCNNPose": GeneralizedRCNNPose}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
