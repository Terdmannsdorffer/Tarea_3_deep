import torch
import numpy as np

def box_iou(boxes1, boxes2):
    """
    Calcula IoU entre dos conjuntos de boxes
    Args:
        boxes1: tensor de forma [N, 4] (x1, y1, x2, y2)
        boxes2: tensor de forma [M, 4] (x1, y1, x2, y2)
    Returns:
        iou: tensor de forma [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Calcular intersección
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def distance_to_bbox(points, bbox, max_dist=None):
    """
    Decodifica distancias a bounding box
    Args:
        points: tensor [N, 2] con coordenadas (x, y)
        bbox: tensor [N, 4] con distancias (left, top, right, bottom)
    Returns:
        boxes: tensor [N, 4] con formato (x1, y1, x2, y2)
    """
    x1 = points[:, 0] - bbox[:, 0]
    y1 = points[:, 1] - bbox[:, 1]
    x2 = points[:, 0] + bbox[:, 2]
    y2 = points[:, 1] + bbox[:, 3]
    
    if max_dist is not None:
        x1 = x1.clamp(min=0, max=max_dist)
        y1 = y1.clamp(min=0, max=max_dist)
        x2 = x2.clamp(min=0, max=max_dist)
        y2 = y2.clamp(min=0, max=max_dist)
    
    return torch.stack([x1, y1, x2, y2], dim=1)

def bbox_to_distance(points, bbox, max_dist=None, eps=1e-6):
    """
    Codifica bounding box como distancias desde puntos
    Args:
        points: [N, 2], bbox: [N, 4] en (x1, y1, x2, y2)
    Returns:
        distances: [N, 4] en (l, t, r, b)
    """
    l = (points[:, 0] - bbox[:, 0]).clamp(min=eps)
    t = (points[:, 1] - bbox[:, 1]).clamp(min=eps)
    r = (bbox[:, 2] - points[:, 0]).clamp(min=eps)
    b = (bbox[:, 3] - points[:, 1]).clamp(min=eps)

    if max_dist is not None:
        l = l.clamp(max=max_dist)
        t = t.clamp(max=max_dist)
        r = r.clamp(max=max_dist)
        b = b.clamp(max=max_dist)

    return torch.stack([l, t, r, b], dim=1)


def compute_centerness(bbox_targets, eps=1e-6):
    """
    bbox_targets: [N, 4] con (l, t, r, b)
    returns: [N] centerness ∈ [0, 1]
    """
    l, t, r, b = bbox_targets[:, 0], bbox_targets[:, 1], bbox_targets[:, 2], bbox_targets[:, 3]

    lr_min = torch.min(l, r)
    lr_max = torch.max(l, r).clamp(min=eps)

    tb_min = torch.min(t, b)
    tb_max = torch.max(t, b).clamp(min=eps)

    centerness = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))
    return centerness


def nms(boxes, scores, threshold=0.5):
    """
    Non-Maximum Suppression
    Args:
        boxes: tensor [N, 4] formato (x1, y1, x2, y2)
        scores: tensor [N]
        threshold: IoU threshold
    Returns:
        keep: indices de las boxes a mantener
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Ordenar por score
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # Calcular IoU con el resto
        iou = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Mantener solo las que tienen IoU menor al threshold
        mask = iou <= threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)