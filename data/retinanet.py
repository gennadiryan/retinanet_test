from collections import OrderedDict
from functools import partial
from typing import Any, List, Mapping, Optional

import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import (
    RetinaNet,
    RetinaNetHead,
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,

    RetinaNet_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d


def load_submodule_params(
    src_dict: Mapping[str, Tensor],
    dest_dict: Mapping[str, Tensor],
    submodules: List[str],
) -> nn.Module:
    submodules = [submodule.split('.') for submodule in submodules]
    is_submodule_param = lambda param_name: (lambda names: len([i for i in range(len(names)) if names[:i + 1] in submodules]) > 0)(param_name.split('.'))
    return OrderedDict(list(dest_dict.items()) + [item for item in src_dict.items() if is_submodule_param(item[0])])


class RetinanetModel(object):
    weights = dict(
        v1=RetinaNet_ResNet50_FPN_Weights.COCO_V1,
        v2=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        backbone=ResNet50_Weights.IMAGENET1K_V1,
    )
    num_classes = 91

    @classmethod
    def get_retinanet(
        cls,
        num_classes: int,
        v2: bool = False,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
        trainable_layers: int = 3,
        **kwargs,
    ) -> nn.Module:
        backbone = cls.get_fpn(pretrained_backbone, trainable_layers)
        anchor_generator = cls.get_anchor_generator()
        head = RetinaNetHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes,
            norm_layer=None if not v2 else partial(nn.GroupNorm, 32),
        )
        head.regression_head._loss_type = 'l1' if not v2 else 'giou'

        model = RetinaNet(
            backbone,
            num_classes,
            anchor_generator=anchor_generator,
            head=head,
        )

        if pretrained:
            state_dict = cls.weights['v1' if not v2 else 'v2'].get_state_dict(progress=True)
            if num_classes != cls.num_classes:
                submodules = ['head.classification_head.cls_logits', 'head.regression_head.bbox_reg']
                # submodules = ['head']
                state_dict = load_submodule_params(model.state_dict(), state_dict, submodules)
            model.load_state_dict(state_dict)

        return model

    @classmethod
    def get_fpn(
        cls,
        pretrained: bool,
        trainable_layers: int,
    ) -> nn.Module:
        return _resnet_fpn_extractor(
            resnet50(
                weights=None if not pretrained else RetinanetModel.weights['backbone'],
                norm_layer=nn.BatchNorm2d if not pretrained else FrozenBatchNorm2d
            ),
            5 if not pretrained else trainable_layers,
            returned_layers=[2, 3, 4],
            extra_blocks=LastLevelP6P7(2048, 256),
            norm_layer=None,
        )

    @classmethod
    def get_anchor_generator(cls) -> nn.Module:
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        return anchor_generator
