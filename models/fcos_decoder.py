

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Scale(nn.Module):
    """Módulo para escalar features learnable"""
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
    
    def forward(self, x):
        return x * self.scale

class FCOSHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_convs=2, use_centerness=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_centerness = use_centerness
        
        # Rama de clasificación
        cls_layers = []
        for i in range(num_convs):
            cls_layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            cls_layers.append(nn.GroupNorm(32, in_channels))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_layers)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        
        # Rama de regresión
        bbox_layers = []
        for i in range(num_convs):
            bbox_layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            bbox_layers.append(nn.GroupNorm(32, in_channels))
            bbox_layers.append(nn.ReLU(inplace=True))
        self.bbox_tower = nn.Sequential(*bbox_layers)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        
        # Rama de centerness (si está habilitada)
        if self.use_centerness:
            ctr_layers = []
            for i in range(num_convs):
                ctr_layers.append(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
                )
                ctr_layers.append(nn.GroupNorm(32, in_channels))
                ctr_layers.append(nn.ReLU(inplace=True))
            self.ctr_tower = nn.Sequential(*ctr_layers)
            self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Inicialización
        self._init_weights()
        
        # Scale para regresión
        self.scales = Scale(1.0)
    
    def _init_weights(self):
        for modules in [self.cls_tower, self.bbox_tower]:
            for layer in modules:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)
        
        if self.use_centerness:
            for layer in self.ctr_tower:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)
        
        # Inicialización especial para las capas finales
        nn.init.normal_(self.cls_logits.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -math.log(99))  # focal loss init
        
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0)
        
        if self.use_centerness:
            nn.init.normal_(self.centerness.weight, mean=0, std=0.01)
            nn.init.constant_(self.centerness.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: tensor [B, C, H, W] - en nuestro caso [B, 256, 16, 16]
        Returns:
            dict con cls_scores, bbox_preds, centerness
        """
        # Rama de clasificación
        cls_tower = self.cls_tower(features)
        cls_scores = self.cls_logits(cls_tower)
        
        # Rama de regresión
        bbox_tower = self.bbox_tower(features)
        bbox_preds = self.scales(self.bbox_pred(bbox_tower))
        bbox_preds = F.relu(bbox_preds)  # Asegurar que las distancias sean positivas
        
        outputs = {
            'cls_scores': cls_scores,      # [B, 2, 16, 16]
            'bbox_preds': bbox_preds        # [B, 4, 16, 16]
        }
        
        # Rama de centerness (si está habilitada)
        if self.use_centerness:
            ctr_tower = self.ctr_tower(features)
            centerness = self.centerness(ctr_tower)
            outputs['centerness'] = centerness  # [B, 1, 16, 16]
        
        return outputs

class FCOSDecoder(nn.Module):
    """
    Decoder FCOS que procesa features de DinoV2
    Implementación fiel a la Figura 1: una sola escala 16x16
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        self.fpn_channels = config.FPN_CHANNELS
        self.use_centerness = config.USE_CENTERNESS
        
        # Proyección de features DinoV2 a dimensión FPN
        self.input_proj = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, self.fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        # FCOS head - una sola escala como en la Figura 1
        self.fcos_head = FCOSHead(
            in_channels=self.fpn_channels,
            num_classes=self.num_classes,
            num_convs=2,  # 2 convoluciones como muestra la figura
            use_centerness=self.use_centerness
        )
        
        # Registro de puntos de anclaje para 16x16
        self._register_anchor_points()
    
    def _register_anchor_points(self):
        """Pre-calcula los puntos de anclaje para la grilla 16x16"""
        # Para imágenes 224x224 con features 16x16, el stride es 14
        stride = 14  # 224 / 16 = 14
        
        # Generar grid de puntos
        y, x = torch.meshgrid(
            torch.arange(16, dtype=torch.float32),
            torch.arange(16, dtype=torch.float32),
            indexing='ij'
        )
        
        # Convertir a coordenadas en la imagen original
        # Centrar los puntos en cada celda
        points = torch.stack([x, y], dim=-1) * stride + stride // 2
        
        self.register_buffer('anchor_points', points)
    
    def forward(self, patch_features):
        """
        Args:
            patch_features: tensor [B, 256, 384] - features de DinoV2
        Returns:
            lista con un único dict (ya que solo hay una escala)
        """
        batch_size = patch_features.shape[0]
        
        # Proyectar features a dimensión FPN
        features = self.input_proj(patch_features)  # [B, 256, 256]
        
        # Reshape a formato espacial (16x16)
        features = features.reshape(batch_size, 16, 16, self.fpn_channels)
        features = features.permute(0, 3, 1, 2)  # [B, 256, 16, 16]
        
        # Aplicar FCOS head
        outputs = self.fcos_head(features)
        
        # Añadir información de la escala
        outputs['level'] = 0
        outputs['stride'] = 14  # 224/16
        outputs['anchor_points'] = self.anchor_points
        
        # Retornar como lista para mantener compatibilidad con el resto del código
        return [outputs]

# models/detector.py mantiene igual, solo procesa una escala ahora

class CatDogDetector(nn.Module):
    """
    Modelo completo que combina DinoV2 (encoder) + FCOS (decoder)
    """
    def __init__(self, config, use_pretrained_encoder=True):
        super().__init__()
        self.config = config
        self.use_pretrained_encoder = use_pretrained_encoder
        
        if use_pretrained_encoder:
            # Cargar DinoV2 solo para inferencia
            self.encoder = None  # Se usarán features pre-calculadas
        else:
            # Cargar DinoV2 para fine-tuning (no recomendado para esta tarea)
            self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            # Congelar encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Decoder FCOS
        self.decoder = FCOSDecoder(config)
    
    def forward(self, inputs):
        """
        Args:
            inputs: Si use_pretrained_encoder=True, son las features pre-calculadas [B, 256, 384]
                   Si no, son las imágenes [B, 3, 224, 224]
        Returns:
            multi_level_outputs: lista de dicts con predicciones para cada escala
        """
        if self.encoder is not None:
            # Extraer features con DinoV2
            with torch.no_grad():
                features = self.encoder.forward_features(inputs)
                patch_features = features['x_norm_patchtokens']
        else:
            # Usar features pre-calculadas
            patch_features = inputs
        
        # Decodificar con FCOS
        outputs = self.decoder(patch_features)
        
        return outputs
    
    def post_process(self, outputs, conf_threshold=0.5, nms_threshold=0.5):
        """
        Post-procesamiento de las predicciones
        Args:
            outputs: salida del modelo (lista de dicts por escala)
            conf_threshold: umbral de confianza
            nms_threshold: umbral para NMS
        Returns:
            detections: lista de dicts con 'boxes', 'scores', 'labels' por imagen
        """
        from utils.box_utils import distance_to_bbox, nms
        
        batch_size = outputs[0]['cls_scores'].shape[0]
        detections = []
        
        for b in range(batch_size):
            batch_boxes = []
            batch_scores = []
            batch_labels = []
            
            # Procesar cada escala
            for level_output in outputs:
                cls_scores = level_output['cls_scores'][b]  # [C, H, W]
                bbox_preds = level_output['bbox_preds'][b]  # [4, H, W]
                anchor_points = level_output['anchor_points']  # [H, W, 2]
                
                if self.config.USE_CENTERNESS and 'centerness' in level_output:
                    centerness = level_output['centerness'][b, 0]  # [H, W]
                else:
                    centerness = torch.ones_like(cls_scores[0])
                
                # Aplicar sigmoid a scores
                cls_scores = cls_scores.sigmoid()
                centerness = centerness.sigmoid()
                
                # Combinar scores con centerness
                cls_scores = cls_scores * centerness.unsqueeze(0)
                
                # Obtener predicciones por encima del umbral
                max_scores, labels = cls_scores.max(dim=0)
                mask = max_scores > conf_threshold
                
                if mask.sum() > 0:
                    # Filtrar predicciones
                    filtered_scores = max_scores[mask]
                    filtered_labels = labels[mask]
                    filtered_bbox_preds = bbox_preds[:, mask].T  # [N, 4]
                    filtered_points = anchor_points[mask]  # [N, 2]
                    
                    # Convertir a bounding boxes
                    boxes = distance_to_bbox(filtered_points, filtered_bbox_preds)
                    
                    batch_boxes.append(boxes)
                    batch_scores.append(filtered_scores)
                    batch_labels.append(filtered_labels)
            
            if len(batch_boxes) > 0:
                # Concatenar todas las detecciones
                all_boxes = torch.cat(batch_boxes, dim=0)
                all_scores = torch.cat(batch_scores, dim=0)
                all_labels = torch.cat(batch_labels, dim=0)
                
                # Aplicar NMS por clase
                keep_indices = []
                for class_id in range(self.config.NUM_CLASSES):
                    class_mask = all_labels == class_id
                    if class_mask.sum() > 0:
                        class_boxes = all_boxes[class_mask]
                        class_scores = all_scores[class_mask]
                        
                        # NMS
                        keep = nms(class_boxes, class_scores, nms_threshold)
                        class_indices = torch.where(class_mask)[0][keep]
                        keep_indices.append(class_indices)
                
                if len(keep_indices) > 0:
                    keep_indices = torch.cat(keep_indices)
                    final_boxes = all_boxes[keep_indices]
                    final_scores = all_scores[keep_indices]
                    final_labels = all_labels[keep_indices]
                else:
                    final_boxes = torch.empty((0, 4))
                    final_scores = torch.empty(0)
                    final_labels = torch.empty(0, dtype=torch.long)
            else:
                final_boxes = torch.empty((0, 4))
                final_scores = torch.empty(0)
                final_labels = torch.empty(0, dtype=torch.long)
            
            detections.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            })
        
        return detections