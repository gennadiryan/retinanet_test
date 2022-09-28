import os
from collections import OrderedDict
from contextlib import ExitStack
from functools import partial
from tqdm import tqdm
from typing import Any, List, Mapping, Optional

import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
# from torchvision.models.detection.retinanet import (
from torchvision_retinanet import (
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

from load_data import VipsDataset
from utils import default_fill
from viz import compute_map, compute_grid, draw


def pad_tensor(tensor: Tensor, modulus: int,) -> Tensor:
    size = list(tensor.size())
    pad_values = [(-s) % modulus if i < 2 and j == 1 else 0 for i, s in enumerate(size[::-1]) for j in range(2)]
    return F.pad(input=tensor, pad=tuple(pad_values), mode='constant', value=0)


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
    defaults = dict(
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        topk_candidates=1000,
    )

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
        default_fill(RetinanetModel.defaults, kwargs)

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
            **kwargs,
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


def train_one_epoch(model, dataloader, device, epoch, progress_bar=False):
    model.train(True)

    losses = list()
    summary = OrderedDict(list())
    bar = None
    if progress_bar:
        bar = tqdm(total=len(dataloader))

    for step, (images, targets) in enumerate(dataloader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]

        loss = model.forward(images, targets)
        losses.append(loss)

        sum(loss.values()).backward()
        optimizer.step()

        for k, v in loss.items():
            summary.setdefault(k, list()).append(v.item())

        disp = OrderedDict([(k, f'{v.item():.2f}') for k, v in loss.items()])

        if progress_bar:
            bar.set_postfix(disp)
            bar.update()
        else:
            print(f'Step: {step}')
            print('\n'.join([f'  {name}: {val}' for name, val in disp.items()]))
            print()

    summary = OrderedDict([(k, f'{torch.tensor(v).mean().item():.2f}') for k, v in summary.items()])

    if progress_bar:
        bar.set_postfix(summary)
        bar.close()
    else:
        print(f'Epoch: {epoch}')
        print('\n'.join([f'  {name}: {val}' for name, val in summary.items()]))
        print()


def evaluate(model, dataset, device, epoch, save_path, metrics_params=None, viz_params=None,):
    model.train(False)
    model_path = os.path.join(save_path, f'model_{epoch}.pt')

    if metrics_params is None:
        metrics_params = dict()
    metrics_path = os.path.join(save_path, f'metrics_{epoch}.pt')

    if viz_params is None:
        viz_params = dict()
    viz_path = os.path.join(save_path, f'viz_{epoch}.png')

    images, targets = tuple(zip(*dataset))
    with torch.no_grad():
        out_targets = [OrderedDict([(k, v.to(torch.device('cpu'))) for k, v in model.forward([image.to(device)])[0].items()]) for image in images]

    metrics = compute_map(targets, out_targets, **metrics_params,)
    grid = compute_grid(images, out_targets, **viz_params,)

    torch.save(model.state_dict(), model_path)
    torch.save(metrics, metrics_path)
    grid.save(viz_path)

    return metrics





if __name__ == '__main__':
    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg'
    vips_img_name = 'XE15-007_1_AmyB_1'
    vips_img_fname = os.path.join(vips_img_dir, f'{vips_img_name}.mrxs')

    json_dir = '/home/gryan/QuPath/v0.3/json'
    json_fnames = [os.path.join(json_dir, subset, f'{vips_img_name}.json') for subset in 'train eval'.split()]

    dataset = VipsDataset(vips_img_fname, json_fnames[0], stride=512, size=1024)
    dataset_test = VipsDataset(vips_img_fname, json_fnames[1], stride=1024, size=1024)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda _: tuple(zip(*_)))

    device = torch.device('cuda', 0)
    train_params = dict(
        optim_params=dict(
            lr=0.0001,
            momentum=0.09,
            weight_decay=0.00001,
        ),
    )
    model_params = dict(
        #Training
        generalized_box_iou=True,

        # Evaluation
        score_thresh=0.1, # this is a good in-between value; 0.05 counts too many spurious detections
        detections_per_img=100, # cap the detections we work with at 100 as there are 8-10 at MOST in this dataset
        batched_nms=False, # helps eliminate multiple predictions per anchor
    )
    run_params = dict(
        ckpt_dir='/home/gryan/projects/rn/artifacts/ckpts',
        ckpt_freq=5,
        run_id='gamma',
        epochs=75,
    )
    metrics_params = dict(
        class_metrics=True,
    )
    viz_params = dict(
        class_names='Core Diffuse Neuritic CAA'.split(),
        class_colors='red green blue yellow'.split(),
    )

    model = RetinanetModel.get_retinanet(4, v2=True, pretrained=True, **model_params)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), **train_params.get('optim_params'))

    ckpt_dir = os.path.join(run_params.get('ckpt_dir'), run_params.get('run_id'))

    if (len(input('Begin training: ')) > 0):
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        for epoch in range(1, run_params.get('epochs') + 1):
            print(f'Epoch: {epoch}')
            print()
            train_one_epoch(model, dataloader, device, epoch, progress_bar=True)
            if epoch == run_params.get('epochs') or epoch % run_params.get('ckpt_freq') == 0:
                print(f'Writing checkpoint to {ckpt_dir}')
                print(evaluate(model, dataset_test, device, epoch, ckpt_dir, metrics_params, viz_params))
                print()
