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
    """
    Cabeza FCOS para clasificación, regresión y centerness
    """
    def __init__(self, in_channels, num_classes, num_convs=4, use_centerness=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_centerness = use_centerness
        
        # Shared convolutions
        cls_subnet = []
        bbox_subnet = []
        
        for i in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU(inplace=True))
            
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU(inplace=True))
        
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        
        # Output layers
        self.cls_score = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        
        if self.use_centerness:
            self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Initialization
        self._init_weights()
        
        # Scales para regresión
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])
    
    def _init_weights(self):
        for modules in [self.cls_subnet, self.bbox_subnet]:
            for layer in modules:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize classification layer with bias
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, -math.log(99))  # focal loss initialization
        
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0)
        
        if self.use_centerness:
            nn.init.normal_(self.centerness.weight, mean=0, std=0.01)
            nn.init.constant_(self.centerness.bias, 0)
    
    def forward(self, features, scale_idx=0):
        """
        Args:
            features: tensor [B, C, H, W]
            scale_idx: índice de la escala para aplicar a la regresión
        Returns:
            dict con cls_scores, bbox_preds, centerness
        """
        cls_feat = self.cls_subnet(features)
        reg_feat = self.bbox_subnet(features)
        
        cls_scores = self.cls_score(cls_feat)
        bbox_preds = self.scales[scale_idx](self.bbox_pred(reg_feat))
        
        # Asegurar que las predicciones de bbox sean positivas
        bbox_preds = F.relu(bbox_preds)
        
        outputs = {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds
        }
        
        if self.use_centerness:
            centerness = self.centerness(reg_feat)
            outputs['centerness'] = centerness
        
        return outputs

class FCOSDecoder(nn.Module):
    """
    Decoder FCOS completo que procesa features de DinoV2
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
        
        # Feature Pyramid Network simplificada
        # Como tenemos patches 16x16, generaremos diferentes escalas
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        
        # Crear capas para diferentes escalas
        for i in range(len(config.STRIDES)):
            if i == 0:
                # Primera escala usa las features directamente
                self.fpn_convs.append(nn.Identity())
            else:
                # Escalas adicionales con convoluciones strided
                layers = []
                for j in range(i):
                    layers.extend([
                        nn.Conv2d(self.fpn_channels, self.fpn_channels, 
                                 kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.fpn_channels),
                        nn.ReLU(inplace=True)
                    ])
                self.fpn_convs.append(nn.Sequential(*layers))
            
            self.fpn_norms.append(nn.GroupNorm(32, self.fpn_channels))
        
        # FCOS head compartida para todas las escalas
        self.fcos_head = FCOSHead(
            in_channels=self.fpn_channels,
            num_classes=self.num_classes,
            use_centerness=self.use_centerness
        )
        
        # Registro de puntos para cada nivel
        self._register_anchor_points()
    
    def _register_anchor_points(self):
        """Pre-calcula los puntos de anclaje para cada nivel de la pirámide"""
        self.anchor_points = {}
        
        # Para patches 16x16 de DinoV2
        base_size = 16
        
        for lvl, stride in enumerate(self.config.STRIDES):
            # Calcular tamaño del feature map en este nivel
            if lvl == 0:
                feat_h = feat_w = base_size
            else:
                feat_h = feat_w = base_size // (2 ** lvl)
            
            # Generar grid de puntos
            y, x = torch.meshgrid(
                torch.arange(feat_h, dtype=torch.float32),
                torch.arange(feat_w, dtype=torch.float32),
                indexing='ij'
            )
            
            # Convertir a coordenadas en la imagen original
            # Ajustar por el stride correspondiente
            points = torch.stack([x, y], dim=-1) * stride + stride // 2
            
            self.anchor_points[lvl] = points
    
    def forward(self, patch_features):
        """
        Args:
            patch_features: tensor [B, 256, 384] - features de DinoV2
        Returns:
            dict con predicciones para cada escala
        """
        batch_size = patch_features.shape[0]
        
        # Proyectar features a dimensión FPN
        features = self.input_proj(patch_features)  # [B, 256, fpn_channels]
        
        # Reshape a formato espacial (16x16)
        features = features.reshape(batch_size, 16, 16, self.fpn_channels)
        features = features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Procesar en múltiples escalas
        multi_level_outputs = []
        
        for lvl, (fpn_conv, fpn_norm) in enumerate(zip(self.fpn_convs, self.fpn_norms)):
            # Aplicar convoluciones para obtener la escala correspondiente
            lvl_features = fpn_conv(features)
            lvl_features = fpn_norm(lvl_features)
            
            # Aplicar FCOS head
            outputs = self.fcos_head(lvl_features, scale_idx=lvl)
            
            # Añadir información del nivel
            outputs['level'] = lvl
            outputs['stride'] = self.config.STRIDES[lvl]
            outputs['anchor_points'] = self.anchor_points[lvl].to(lvl_features.device)
            
            multi_level_outputs.append(outputs)
        
        return multi_level_outputs