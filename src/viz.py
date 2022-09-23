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

def evaluate(model, device, images, top_k=10, **kwargs):
    model.train(False)
    out = model.forward([image.to(device) for image in images])
    return [draw(*pair, top_k=top_k, **kwargs) for pair in zip(images, out)]

def evaluate_grid(model, device, dataset, **kwargs):
    return torchvision.transforms.ToPILImage()(torchvision.utils.make_grid([evaluate(model, device, [pair[0]], **kwargs)[0] for pair in dataset], int(sqrt(len(dataset)))))

def reference_grid(dataset, **kwargs):
    return torchvision.transforms.ToPILImage()(torchvision.utils.make_grid([draw(*pair, top_k=len(pair[1]['labels']), **kwargs) for pair in dataset], int(sqrt(len(dataset)))))
