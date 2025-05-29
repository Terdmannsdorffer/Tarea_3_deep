import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import box_iou, bbox_to_distance, compute_centerness

class FocalLoss(nn.Module):
    """Focal Loss para manejar desbalance de clases"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits sin sigmoid
            targets: [N] etiquetas de clase
        """
        p = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()

class IoULoss(nn.Module):
    """IoU Loss para regresión de bounding boxes"""
    def __init__(self, loss_type='iou'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, 4] predicciones de distancias (l, t, r, b)
            target: [N, 4] targets de distancias
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]
        
        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)
        
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_top, target_top) + torch.min(pred_bottom, target_bottom)
        
        area_intersect = w_intersect * h_intersect
        area_union = pred_area + target_area - area_intersect
        
        iou = area_intersect / area_union.clamp(min=1e-6)
        
        if self.loss_type == 'iou':
            loss = 1 - iou
        elif self.loss_type == 'giou':
            # GIoU loss
            w_enclose = torch.max(pred_left + pred_right, target_left + target_right)
            h_enclose = torch.max(pred_top + pred_bottom, target_top + target_bottom)
            area_enclose = w_enclose * h_enclose
            giou = iou - (area_enclose - area_union) / area_enclose.clamp(min=1e-6)
            loss = 1 - giou
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss.mean()

class FCOSLoss(nn.Module):
    """Loss completa para FCOS"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.focal_loss = FocalLoss()
        self.iou_loss = IoULoss(loss_type='giou')
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: lista de dicts con predicciones por escala
            targets: lista de dicts con 'boxes' y 'labels' por imagen
        """
        cls_losses = []
        reg_losses = []
        centerness_losses = []
        
        batch_size = len(targets['boxes'])
        
        for level_idx, level_output in enumerate(outputs):
            cls_scores = level_output['cls_scores']  # [B, C, H, W]
            bbox_preds = level_output['bbox_preds']  # [B, 4, H, W]
            anchor_points = level_output['anchor_points']  # [H, W, 2]
            stride = level_output['stride']
            
            # Obtener dispositivo del modelo
            device = cls_scores.device
            
            # Flatten predictions
            B, C, H, W = cls_scores.shape
            cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            bbox_preds = bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)  # [B*H*W, 4]
            
            if self.config.USE_CENTERNESS and 'centerness' in level_output:
                centerness = level_output['centerness'].permute(0, 2, 3, 1).reshape(-1)  # [B*H*W]
            
            # Asignar targets a cada punto
            cls_targets = []
            reg_targets = []
            centerness_targets = []
            
            for b in range(batch_size):
                gt_boxes = targets['boxes'][b].to(cls_scores.device)  # [N, 4] - Mover al mismo device que el modelo
                gt_labels = targets['labels'][b].to(cls_scores.device)  # [N] - Mover al mismo device que el modelo
                
                if len(gt_boxes) == 0:
                    # No hay objetos en esta imagen
                    cls_target = torch.zeros((H * W, C), device=cls_scores.device)
                    reg_target = torch.zeros((H * W, 4), device=bbox_preds.device)
                    centerness_target = torch.zeros(H * W, device=cls_scores.device)
                else:
                    # Asignar cada punto al GT más cercano
                    points = anchor_points.reshape(-1, 2)  # [H*W, 2]
                    
                    # Calcular distancias a todos los GT boxes
                    num_points = points.shape[0]
                    num_gts = gt_boxes.shape[0]
                    
                    # Expandir para broadcasting
                    points_exp = points.unsqueeze(1).expand(num_points, num_gts, 2)
                    
                    # Verificar si cada punto está dentro de cada GT box
                    inside_gt = (points_exp[:, :, 0] >= gt_boxes[:, 0]) & \
                               (points_exp[:, :, 0] <= gt_boxes[:, 2]) & \
                               (points_exp[:, :, 1] >= gt_boxes[:, 1]) & \
                               (points_exp[:, :, 1] <= gt_boxes[:, 3])
                    
                    # Para puntos dentro de múltiples GT, elegir el más pequeño
                    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                    areas = areas.unsqueeze(0).expand(num_points, num_gts)
                    
                    # Clonar para evitar warning y asignar área infinita a puntos fuera del GT
                    areas = areas.clone()
                    areas[~inside_gt] = float('inf')
                    
                    # Encontrar el GT con área mínima para cada punto
                    min_areas, min_area_idx = areas.min(dim=1)
                    
                    # Crear targets
                    cls_target = torch.zeros((num_points, C), device=cls_scores.device)
                    reg_target = torch.zeros((num_points, 4), device=bbox_preds.device)
                    centerness_target = torch.zeros(num_points, device=cls_scores.device)
                    
                    # Máscara de puntos positivos
                    pos_mask = min_areas < float('inf')
                    pos_indices = torch.where(pos_mask)[0]
                    
                    if len(pos_indices) > 0:
                        # Asignar clase
                        assigned_labels = gt_labels[min_area_idx[pos_indices]]
                        cls_target[pos_indices, assigned_labels] = 1
                        
                        # Calcular targets de regresión
                        assigned_boxes = gt_boxes[min_area_idx[pos_indices]]
                        pos_points = points[pos_indices]
                        
                        # Convertir a formato de distancias
                        reg_target[pos_indices] = bbox_to_distance(pos_points, assigned_boxes, 
                                                                   max_dist=stride * 8)
                        
                        # Calcular centerness
                        if self.config.USE_CENTERNESS:
                            centerness_target[pos_indices] = compute_centerness(reg_target[pos_indices])
                
                cls_targets.append(cls_target)
                reg_targets.append(reg_target)
                centerness_targets.append(centerness_target)
            
            # Stack targets
            cls_targets = torch.stack(cls_targets).reshape(-1, C)  # [B*H*W, C]
            reg_targets = torch.stack(reg_targets).reshape(-1, 4)  # [B*H*W, 4]
            
            # Calcular losses
            pos_mask = cls_targets.sum(dim=1) > 0
            num_pos = pos_mask.sum().item()
            
            # Classification loss
            cls_loss = self.focal_loss(cls_scores, cls_targets)
            cls_losses.append(cls_loss)
            
            if num_pos > 0:
                # Regression loss (solo para positivos)
                reg_loss = self.iou_loss(bbox_preds[pos_mask], reg_targets[pos_mask])
                reg_losses.append(reg_loss)
                
                # Centerness loss
                if self.config.USE_CENTERNESS:
                    centerness_targets = torch.stack(centerness_targets).reshape(-1)
                    centerness_loss = self.bce_loss(centerness[pos_mask], 
                                                   centerness_targets[pos_mask])
                    centerness_losses.append(centerness_loss)
        
        # Combinar losses
        total_cls_loss = sum(cls_losses) / len(cls_losses)
        total_reg_loss = sum(reg_losses) / len(reg_losses) if reg_losses else torch.tensor(0.0)
        total_centerness_loss = sum(centerness_losses) / len(centerness_losses) if centerness_losses else torch.tensor(0.0)
        
        total_loss = (self.config.CLS_LOSS_WEIGHT * total_cls_loss + 
                     self.config.REG_LOSS_WEIGHT * total_reg_loss +
                     self.config.CENTERNESS_LOSS_WEIGHT * total_centerness_loss)
        
        return {
            'total_loss': total_loss,
            'cls_loss': total_cls_loss,
            'reg_loss': total_reg_loss,
            'centerness_loss': total_centerness_loss
        }