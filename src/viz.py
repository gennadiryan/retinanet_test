import os
from math import sqrt

import torch
import torchvision


def draw(image, target, top_k=10, class_names=None, class_colors=None,):
    image = (image * 255).to(torch.uint8)
    labels, boxes = tuple(target.get(key)[:top_k] for key in 'labels boxes'.split())
    names = [f'Class {label}' for label in labels]
    if class_names is not None:
        names = [class_names[label] for label in (labels - 1)]
    if 'scores' in target.keys():
        names = [f'{name} ({int(score * 100) / 100})' for name, score in zip(names, target.get('scores')[:top_k])]
    colors = ['black'] * len(labels)
    if class_colors is not None:
        colors = [class_colors[label] for label in (labels - 1)]
    return torchvision.utils.draw_bounding_boxes(image, boxes, names, colors)


def evaluate_iou(model, images, targets, top_k=10, **kwargs):
    model.to(torch.device('cpu'))
    model.train(False)
    out_targets = [model.forward([image])[0] for image in images]
    images = [draw(*pair, top_k=top_k, **kwargs) for pair in zip(images, out_targets)]
    ious = [torchvision.ops.generalized_box_iou(target['boxes'], out_target['boxes']) for target, out_target in zip(targets, out_targets)]
    return images, out_targets, ious

def evaluate_iou_grid(model, dataset, **kwargs):
    images, targets, ious = evaluate_iou(model, *tuple(zip(*dataset)), **kwargs)
    return torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(images, int(sqrt(len(dataset))))), targets, ious

def reference_grid(dataset, **kwargs):
    return torchvision.transforms.ToPILImage()(torchvision.utils.make_grid([draw(*pair, top_k=len(pair[1]['labels']), **kwargs) for pair in dataset], int(sqrt(len(dataset)))))

def evaluate_checkpoint(model, dataset, epoch, save_path, **kwargs):
    grid_path, model_path, out_path = [os.path.join(save_path, name) for name in (f'grid_{epoch}.png', f'model_{epoch}.pt', f'out_{epoch}.pt')]
    grid, outputs, ious = evaluate_iou_grid(model, dataset, **kwargs)
    grid.save(grid_path)
    torch.save(model.state_dict(), model_path)
    torch.save(outputs, out_path)
