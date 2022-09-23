import os
import json
import numpy as np
import pyvips

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import draw_bounding_boxes

from utils import encode, decode, bboxes_to_tiles_map



class JsonDataset(Dataset):
    def __init__(self, json_name, stride=1024, size=1024, label_names='Core Diffuse Neuritic CAA'.split()):
        self.stride, self.size = stride, size
        self.labels, self.bboxes = [np.array(_) for _ in zip(*tuple(map(lambda json_obj: JsonDataset.json_to_instance(json_obj, label_names), json.loads(open(json_name, 'r').read()).get('annotations'))))]
        self.tiles = bboxes_to_tiles_map(self.bboxes, self.stride, self.size)

    @staticmethod
    def json_to_instance(json_obj, label_names):
        return tuple(np.array(_, dtype=np.intc) for _ in (1 + label_names.index(json_obj.get('pathClasses')[0]), ([f(coords) for f in [np.amin, np.amax] for coords in [np.array(json_obj.get(c)).astype(np.intc) for c in 'x y'.split()]])))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = list(self.tiles.keys())[idx]
        tile_min = np.array(tile) * self.stride
        tile_max = tile_min + self.size

        idxs = self.tiles.get(tile)
        labels, bboxes = self.labels[idxs], self.bboxes[idxs]
        bboxes[:, :2] = np.maximum(tile_min, bboxes[:, :2]) - tile_min
        bboxes[:, 2:] = np.minimum(tile_max, bboxes[:, 2:]) - tile_min

        return dict(labels=torch.as_tensor(labels.astype(np.int_)), boxes=torch.as_tensor(bboxes.astype(np.int_)))

class VipsDataset(JsonDataset):
    def __init__(self, vips_img_name, json_name, **kwargs):
        super().__init__(json_name, **kwargs)
        self.vips_img = pyvips.Image.new_from_file(vips_img_name, level=0)
        self.transform = ToTensor()

    @staticmethod
    def vips_crop(vips_img, x, y, w, h):
        x, y = tuple(map(sum, zip((x, y), [int(vips_img.get(f'openslide.bounds-{coord}')) for coord in 'x y'.split()])))
        vips_crop = vips_img[:3].crop(x, y, w, h)
        return np.ndarray(buffer=vips_crop.write_to_memory(), dtype=np.uint8, shape=(vips_crop.height, vips_crop.width, vips_crop.bands))

    def __getitem__(self, idx):
        tile = [_ * self.stride for _ in list(self.tiles.keys())[idx]]
        return self.transform(VipsDataset.vips_crop(self.vips_img, *tile, *([self.size] * 2))), super().__getitem__(idx)


if __name__ == '__main__':
    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg'
    vips_img_name = 'XE15-007_1_AmyB_1'
    vips_img_fname = os.path.join(vips_img_dir, f'{vips_img_name}.mrxs')

    json_dir = '/home/gryan/QuPath/v0.3/json'
    json_fname = os.path.join(json_dir, f'{vips_img_name}.json')

    dataset = VipsDataset(vips_img_fname, json_fname)

    to_pil = ToPILImage()
    for image, target in dataset:
        if len(input('Show next tile: ')) > 0:
            break
        image = draw_bounding_boxes(image, target['boxes'], colors='black')
        to_pil(image).show()
