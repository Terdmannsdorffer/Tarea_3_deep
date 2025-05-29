import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    running_centerness_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Mover datos al dispositivo
        features = batch['features'].to(device)
        
        # Forward pass
        outputs = model(features)
        
        # Calcular loss
        losses = criterion(outputs, batch)
        loss = losses['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Actualizar estadísticas
        running_loss += loss.item()
        running_cls_loss += losses['cls_loss'].item()
        running_reg_loss += losses['reg_loss'].item()
        running_centerness_loss += losses['centerness_loss'].item()
        
        # Actualizar barra de progreso
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'cls': running_cls_loss / (batch_idx + 1),
            'reg': running_reg_loss / (batch_idx + 1),
            'ctr': running_centerness_loss / (batch_idx + 1)
        })
    
    return running_loss / len(dataloader)

def main():
    # Importar aquí para evitar importaciones circulares
    from config import Config
    from dataset import CatDogDetectionDataset, collate_fn
    from models.detector import CatDogDetector
    from utils.losses import FCOSLoss
    from utils.metrics import evaluate_map
    
    # Configuración
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Datasets y dataloaders
    train_dataset = CatDogDetectionDataset(config, split='train', use_precomputed_features=True)
    val_dataset = CatDogDetectionDataset(config, split='val', use_precomputed_features=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Modelo
    model = CatDogDetector(config, use_pretrained_encoder=True).to(device)
    
    # Loss y optimizador
    criterion = FCOSLoss(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    # Directorio para checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    best_map = 0.0
    
    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Entrenar
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Evaluar
        if epoch % 5 == 0:
            print(f'\nEvaluating at epoch {epoch}...')
            results = evaluate_map(model, val_loader, config, device)
            
            print(f"Epoch {epoch} - mAP: {results['mAP']:.4f}")
            for class_name, ap in zip(config.CLASSES, results['AP']):
                print(f"  {class_name}: {ap:.4f}")
            
            # Guardar mejor modelo
            if results['mAP'] > best_map:
                best_map = results['mAP']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mAP': best_map,
                    'config': config
                }, f'checkpoints/best_model.pth')
                print(f'Saved best model with mAP: {best_map:.4f}')
        
        # Guardar checkpoint regular
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
        
        scheduler.step()
    
    print(f'\nTraining completed! Best mAP: {best_map:.4f}')

if __name__ == '__main__':
    main()