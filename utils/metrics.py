import torch
import numpy as np
from collections import defaultdict
from utils.box_utils import box_iou

def calculate_ap(recall, precision):
    """
    Calcula Average Precision usando interpolación de 11 puntos
    """
    # Interpolación de 11 puntos
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    return ap

def evaluate_map(model, dataloader, config, device):
    """
    Evalúa el modelo y calcula mAP
    """
    model.eval()
    
    # Almacenar todas las predicciones y GT
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            
            # Obtener predicciones
            outputs = model(features)
            detections = model.post_process(outputs, 
                                           conf_threshold=config.CONF_THRESHOLD,
                                           nms_threshold=config.NMS_THRESHOLD)
            
            # Procesar cada imagen del batch
            for i, (det, gt_boxes, gt_labels) in enumerate(zip(detections, 
                                                               batch['boxes'], 
                                                               batch['labels'])):
                # Guardar predicciones
                if len(det['boxes']) > 0:
                    for box, score, label in zip(det['boxes'], det['scores'], det['labels']):
                        all_predictions[label.item()].append({
                            'box': box.cpu().numpy(),
                            'score': score.item(),
                            'image_idx': 1
                        })
                
                # Guardar ground truth
                for box, label in zip(gt_boxes, gt_labels):
                    all_ground_truths[label.item()].append({
                        'box': box.cpu().numpy(),
                        'image_idx': 1
                    })
    
    # Calcular AP para cada clase
    aps = []
    for class_id in range(config.NUM_CLASSES):
        # Obtener predicciones y GT para esta clase
        predictions = all_predictions[class_id]
        ground_truths = all_ground_truths[class_id]
        
        if len(ground_truths) == 0:
            aps.append(0)
            continue
        
        if len(predictions) == 0:
            aps.append(0)
            continue
        
        # Ordenar predicciones por score
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        # Calcular TP y FP
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        # Marcar GT como no detectados
        gt_detected = defaultdict(set)
        
        for pred_idx, pred in enumerate(predictions):
            pred_box = pred['box']
            pred_img_idx = pred['image_idx']
            
            # Obtener GT de la misma imagen
            img_gts = [gt for gt in ground_truths if gt['image_idx'] == pred_img_idx]
            
            if len(img_gts) == 0:
                fp[pred_idx] = 1
                continue
            
            # Calcular IoU con todos los GT
            ious = []
            for gt_idx, gt in enumerate(img_gts):
                iou = calculate_iou(pred_box, gt['box'])
                ious.append((iou, gt_idx))
            
            # Encontrar mejor match
            ious.sort(key=lambda x: x[0], reverse=True)
            best_iou, best_gt_idx = ious[0]
            
            # Verificar si es TP o FP (threshold IoU = 0.5)
            if best_iou >= 0.5 and best_gt_idx not in gt_detected[pred_img_idx]:
                tp[pred_idx] = 1
                gt_detected[pred_img_idx].add(best_gt_idx)
            else:
                fp[pred_idx] = 1
        
        # Calcular precision y recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calcular AP
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
    
    # Calcular mAP
    mAP = np.mean(aps)
    
    return {
        'mAP': mAP,
        'AP': aps
    }

def calculate_iou(box1, box2):
    """Calcula IoU entre dos boxes en formato [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0