# main_fixed.py - Script principal corregido

import os
import sys
import torch
import argparse
from datetime import datetime
import shutil

# Verificar CUDA primero
if not torch.cuda.is_available():
    print("⚠️  ADVERTENCIA: CUDA no está disponible. El entrenamiento será MUY lento en CPU.")
    print("Verifica tu instalación de PyTorch con: python -c \"import torch; print(torch.cuda.is_available())\"")
    response = input("¿Continuar de todos modos? (s/n): ")
    if response.lower() != 's':
        sys.exit(1)

# Importar módulos del proyecto
from config import Config
from preprocess_features import precompute_dino_features
from dataset import CatDogDetectionDataset, collate_fn
from models.detector import CatDogDetector
from utils.losses import FCOSLoss
from utils.metrics import evaluate_map
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

def setup_experiment(experiment_name, config):
    """Configura directorios para el experimento"""
    # Usar directorio más simple
    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/visualizations", exist_ok=True)
    
    # Guardar configuración
    with open(f"{exp_dir}/config.txt", 'w') as f:
        for attr in dir(config):
            if not attr.startswith('__'):
                value = getattr(config, attr)
                f.write(f"{attr}: {value}\n")
    
    return exp_dir

def train_with_config(config, exp_dir, device):
    """Entrena el modelo con la configuración dada"""
    # Datasets
    print("  Cargando datasets...")
    train_dataset = CatDogDetectionDataset(config, split='train', use_precomputed_features=True)
    val_dataset = CatDogDetectionDataset(config, split='val', use_precomputed_features=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0 if device.type == 'cpu' else 4,  # 0 workers en CPU
        pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=0 if device.type == 'cpu' else 4,
        pin_memory=device.type == 'cuda'
    )
    
    # Modelo
    model = CatDogDetector(config, use_pretrained_encoder=True).to(device)
    print(f"  Parámetros entrenables: {model.get_num_parameters():,}")
    
    # Loss y optimizador
    criterion = FCOSLoss(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    best_map = 0.0
    checkpoint_path = f"{exp_dir}/checkpoints/best_model.pth"
    
    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Entrenar
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'  Época {epoch}/{config.NUM_EPOCHS}')
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(device)
            outputs = model(features)
            losses = criterion(outputs, batch)
            loss = losses['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
        # Evaluar SIEMPRE en la última época o cada 5 épocas
        if epoch % 5 == 0 or epoch == config.NUM_EPOCHS:
            print(f"    Evaluando época {epoch}...")
            results = evaluate_map(model, val_loader, config, device)
            print(f"    Época {epoch} - mAP: {results['mAP']:.4f}")
            
            # Guardar si es mejor O si es la última época
            if results['mAP'] > best_map or epoch == config.NUM_EPOCHS:
                if results['mAP'] > best_map:
                    best_map = results['mAP']
                
                # Guardar checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mAP': results['mAP'],
                    'config': config
                }, checkpoint_path)
                print(f"    💾 Modelo guardado (mAP: {results['mAP']:.4f})")
        
        scheduler.step()
    
    # Si no se guardó ningún modelo, guardar el último
    if not os.path.exists(checkpoint_path):
        print("    ⚠️  Guardando modelo final...")
        torch.save({
            'epoch': config.NUM_EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mAP': 0.0,
            'config': config
        }, checkpoint_path)
    
    print(f"  ✓ Entrenamiento completado. Mejor mAP: {best_map:.4f}")
    return checkpoint_path

def evaluate_model(checkpoint_path, config, device):
    """Evalúa un modelo desde un checkpoint"""
    # Verificar que existe el checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ No se encuentra el checkpoint: {checkpoint_path}")
        return {'mAP': 0.0, 'AP': [0.0, 0.0]}
    
    # Dataset de validación
    val_dataset = CatDogDetectionDataset(config, split='val', use_precomputed_features=True)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=0 if device.type == 'cpu' else 4
    )
    
    # Cargar modelo
    model = CatDogDetector(config, use_pretrained_encoder=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluar
    results = evaluate_map(model, val_loader, config, device)
    return results

def run_complete_pipeline(args):
    """Ejecuta el pipeline completo de entrenamiento y evaluación"""
    
    print("="*60)
    print("PIPELINE COMPLETO DE DETECCIÓN DE PERROS Y GATOS")
    print("="*60)
    
    # Configuración
    config = Config()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    # Ajustar configuración para CPU si es necesario
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("\n⚠️  Ejecutando en CPU - Ajustando configuración...")
        config.BATCH_SIZE = min(config.BATCH_SIZE, 16)  # Reducir batch size
        config.NUM_WORKERS = 0  # Sin multiprocessing en CPU
        config.PIN_MEMORY = False
    
    print(f"\n✓ Dispositivo: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Paso 1: Verificar dataset
    print(f"\n1️⃣ Verificando dataset en: {config.DATA_ROOT}")
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: No se encuentra el dataset en {config.DATA_ROOT}")
        return
    
    # Verificar estructura
    required_files = [
        config.IMAGES_DIR,
        config.ANNOTATIONS_DIR,
        config.TRAIN_FILE,
        config.VAL_FILE
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {os.path.basename(file_path)}")
        else:
            print(f"  ❌ Falta: {file_path}")
            return
    
    # Paso 2: Pre-calcular features de DinoV2
    print(f"\n2️⃣ Pre-calculando features de DinoV2...")
    if not os.path.exists(config.FEATURES_DIR) or args.force_features:
        print("  Extrayendo features (esto puede tomar varios minutos)...")
        precompute_dino_features(config)
        print("  ✓ Features extraídas y guardadas")
    else:
        print("  ✓ Features ya existen, saltando extracción")
    
    # Experimento 1: CON Centerness
    if args.experiment in ['both', 'with_centerness']:
        print(f"\n3️⃣ EXPERIMENTO 1: Entrenamiento CON Centerness")
        print("-"*50)
        
        config.USE_CENTERNESS = True
        exp_dir_with = setup_experiment("with_centerness", config)
        
        print(f"  Configuración:")
        print(f"    - Épocas: {config.NUM_EPOCHS}")
        print(f"    - Batch size: {config.BATCH_SIZE}")
        print(f"    - Learning rate: {config.LEARNING_RATE}")
        print(f"    - Centerness: {config.USE_CENTERNESS}")
        print(f"    - Dispositivo: {device}")
        
        # Entrenar modelo con centerness
        checkpoint_with = train_with_config(config, exp_dir_with, device)
        
        # Evaluar
        print(f"\n  Evaluando modelo CON centerness...")
        results_with = evaluate_model(checkpoint_with, config, device)
        
        # Guardar resultados
        with open(f"{exp_dir_with}/results.txt", 'w') as f:
            f.write(f"mAP: {results_with['mAP']:.4f}\n")
            for class_name, ap in zip(config.CLASSES, results_with['AP']):
                f.write(f"{class_name}: {ap:.4f}\n")
        
        print(f"\n  Resultados CON centerness:")
        print(f"    mAP: {results_with['mAP']:.4f}")
        for class_name, ap in zip(config.CLASSES, results_with['AP']):
            print(f"    {class_name}: {ap:.4f}")
    
    # Experimento 2: SIN Centerness
    if args.experiment in ['both', 'without_centerness']:
        print(f"\n4️⃣ EXPERIMENTO 2: Entrenamiento SIN Centerness")
        print("-"*50)
        
        config.USE_CENTERNESS = False
        exp_dir_without = setup_experiment("without_centerness", config)
        
        print(f"  Configuración:")
        print(f"    - Épocas: {config.NUM_EPOCHS}")
        print(f"    - Batch size: {config.BATCH_SIZE}")
        print(f"    - Learning rate: {config.LEARNING_RATE}")
        print(f"    - Centerness: {config.USE_CENTERNESS}")
        
        # Entrenar modelo sin centerness
        checkpoint_without = train_with_config(config, exp_dir_without, device)
        
        # Evaluar
        print(f"\n  Evaluando modelo SIN centerness...")
        results_without = evaluate_model(checkpoint_without, config, device)
        
        # Guardar resultados
        with open(f"{exp_dir_without}/results.txt", 'w') as f:
            f.write(f"mAP: {results_without['mAP']:.4f}\n")
            for class_name, ap in zip(config.CLASSES, results_without['AP']):
                f.write(f"{class_name}: {ap:.4f}\n")
        
        print(f"\n  Resultados SIN centerness:")
        print(f"    mAP: {results_without['mAP']:.4f}")
        for class_name, ap in zip(config.CLASSES, results_without['AP']):
            print(f"    {class_name}: {ap:.4f}")
    
    # Comparación final
    if args.experiment == 'both' and 'results_with' in locals() and 'results_without' in locals():
        print(f"\n5️⃣ COMPARACIÓN FINAL")
        print("="*60)
        print(f"CON Centerness - mAP: {results_with['mAP']:.4f}")
        print(f"SIN Centerness - mAP: {results_without['mAP']:.4f}")
        print(f"Diferencia: {results_with['mAP'] - results_without['mAP']:.4f}")
        if results_without['mAP'] > 0:
            print(f"Mejora porcentual: {((results_with['mAP'] - results_without['mAP']) / results_without['mAP'] * 100):.2f}%")
    
    print(f"\n✅ Pipeline completado!")

def main():
    parser = argparse.ArgumentParser(description='Pipeline completo de detección de perros y gatos')
    parser.add_argument('--experiment', type=str, default='both',
                       choices=['both', 'with_centerness', 'without_centerness'],
                       help='Qué experimento ejecutar (default: both)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Número de épocas de entrenamiento (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Tamaño del batch (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--force-features', action='store_true',
                       help='Forzar recálculo de features de DinoV2')
    parser.add_argument('--quick-test', action='store_true',
                       help='Modo de prueba rápida (5 épocas)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 5
        print("🚀 MODO DE PRUEBA RÁPIDA: Solo 5 épocas")
    
    run_complete_pipeline(args)

if __name__ == '__main__':
    main()