# models/detector.py - Modelo completo de detección

import torch
import torch.nn as nn
from models.fcos_decoder import FCOSDecoder
from utils.box_utils import distance_to_bbox, nms

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
                    
                    # Clamp boxes to image boundaries
                    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=self.config.IMAGE_SIZE)
                    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=self.config.IMAGE_SIZE)
                    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=self.config.IMAGE_SIZE)
                    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=self.config.IMAGE_SIZE)
                    
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
    
    def predict(self, image_features, conf_threshold=0.5, nms_threshold=0.5):
        """
        Método de conveniencia para predicción en una sola imagen
        Args:
            image_features: features de una imagen [256, 384] o [1, 256, 384]
            conf_threshold: umbral de confianza
            nms_threshold: umbral para NMS
        Returns:
            dict con 'boxes', 'scores', 'labels'
        """
        # Asegurar dimensión de batch
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(image_features)
            detections = self.post_process(outputs, conf_threshold, nms_threshold)
        
        return detections[0]  # Retornar solo la primera imagen
    
    def load_checkpoint(self, checkpoint_path, device='cuda'):
        """
        Cargar modelo desde checkpoint
        Args:
            checkpoint_path: ruta al archivo .pth
            device: dispositivo donde cargar el modelo
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo cargado desde {checkpoint_path}")
        
        if 'mAP' in checkpoint:
            print(f"mAP del checkpoint: {checkpoint['mAP']:.4f}")
        if 'epoch' in checkpoint:
            print(f"Época del checkpoint: {checkpoint['epoch']}")
    
    def get_num_parameters(self, trainable_only=True):
        """
        Obtener número de parámetros del modelo
        Args:
            trainable_only: si True, solo cuenta parámetros entrenables
        Returns:
            número de parámetros
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def freeze_encoder(self):
        """Congela los parámetros del encoder (si existe)"""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder congelado")
    
    def unfreeze_encoder(self):
        """Descongela los parámetros del encoder (si existe)"""
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("Encoder descongelado")
    
    def get_optimizer_groups(self, lr_encoder=1e-5, lr_decoder=1e-3):
        """
        Obtiene grupos de parámetros con diferentes learning rates
        Útil si se quiere hacer fine-tuning del encoder con un LR menor
        Args:
            lr_encoder: learning rate para el encoder
            lr_decoder: learning rate para el decoder
        Returns:
            lista de grupos de parámetros para el optimizador
        """
        if self.encoder is not None:
            encoder_params = list(self.encoder.parameters())
            decoder_params = list(self.decoder.parameters())
            
            return [
                {'params': encoder_params, 'lr': lr_encoder},
                {'params': decoder_params, 'lr': lr_decoder}
            ]
        else:
            return [{'params': self.decoder.parameters(), 'lr': lr_decoder}]