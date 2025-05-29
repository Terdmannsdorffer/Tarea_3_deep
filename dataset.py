import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class CatDogDetectionDataset(Dataset):
    def __init__(self, config, split='train', use_precomputed_features=True):
        self.config = config
        self.split = split
        self.use_precomputed_features = use_precomputed_features
        
        # Cargar lista de archivos
        split_file = config.TRAIN_FILE if split == 'train' else config.VAL_FILE
        with open(split_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
        
        # Transformaciones para imágenes
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # Cargar anotaciones
        annotation = self.load_annotation(file_name)
        
        if self.use_precomputed_features:
            # Cargar features pre-calculadas
            features_path = os.path.join(self.config.FEATURES_DIR, f"{file_name}.pt")
            features = torch.load(features_path)
        else:
            # Cargar y transformar imagen
            img_path = os.path.join(self.config.IMAGES_DIR, f"{file_name}.png")
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            features = image
        
        return {
            'features': features,
            'boxes': annotation['boxes'],
            'labels': annotation['labels'],
            'file_name': file_name,
            'original_size': annotation['original_size']
        }
    
    def load_annotation(self, file_name):
        xml_path = os.path.join(self.config.ANNOTATIONS_DIR, f"{file_name}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Obtener tamaño original
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            label = self.config.CLASS_TO_IDX[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Normalizar coordenadas al tamaño de la imagen redimensionada
            xmin = xmin * self.config.IMAGE_SIZE / width
            ymin = ymin * self.config.IMAGE_SIZE / height
            xmax = xmax * self.config.IMAGE_SIZE / width
            ymax = ymax * self.config.IMAGE_SIZE / height
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'original_size': (width, height)
        }

def collate_fn(batch):
    """Función para manejar batches con número variable de objetos"""
    features = torch.stack([item['features'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'features': features,
        'boxes': boxes,
        'labels': labels,
        'file_names': file_names,
        'original_sizes': original_sizes
    }