import torch
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from config import Config

def precompute_dino_features(config):
    """Pre-calcula y guarda las features de DinoV2 para todas las imágenes"""
    
    # Crear directorio para features
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar modelo DinoV2
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    model.eval()
    
    # Transformaciones
    preprocess = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Obtener lista de todas las imágenes
    all_files = []
    for split_file in [config.TRAIN_FILE, config.VAL_FILE]:
        with open(split_file, 'r') as f:
            all_files.extend([line.strip() for line in f.readlines()])
    
    # Procesar en batches
    batch_size = 32
    for i in tqdm(range(0, len(all_files), batch_size), desc="Extracting features"):
        batch_files = all_files[i:i+batch_size]
        batch_images = []
        
        # Cargar imágenes del batch
        for file_name in batch_files:
            img_path = os.path.join(config.IMAGES_DIR, f"{file_name}.png")
            image = Image.open(img_path).convert('RGB')
            image = preprocess(image)
            batch_images.append(image)
        
        # Stack y procesar batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            features = model.forward_features(batch_tensor)
            patch_tokens = features['x_norm_patchtokens']  # [batch_size, 256, 384]
        
        # Guardar features individuales
        for j, file_name in enumerate(batch_files):
            feature_path = os.path.join(config.FEATURES_DIR, f"{file_name}.pt")
            torch.save(patch_tokens[j].cpu(), feature_path)
    
    print(f"Features saved to {config.FEATURES_DIR}")

if __name__ == "__main__":
    config = Config()
    precompute_dino_features(config)