import torch
import argparse
from torch.utils.data import DataLoader
from config import Config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os

def visualize_predictions(model, dataset, config, device, num_samples=10):
    """Visualiza predicciones del modelo"""
    model.eval()
    
    # Crear directorio para guardar visualizaciones
    os.makedirs('visualizations', exist_ok=True)
    
    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    good_examples = []
    bad_examples = []
    
    with torch.no_grad():
        for idx in indices:
            data = dataset[idx]
            features = data['features'].unsqueeze(0).to(device)
            
            # Predicciones
            outputs = model(features)
            for lvl_out in outputs:
                print(f"Nivel {lvl_out['level']}:")
                print("  Max score:", lvl_out['cls_scores'].sigmoid().max().item())
                print("  Centerness:", lvl_out['centerness'].sigmoid().mean().item() if 'centerness' in lvl_out else "N/A")
                print("  BBox mean:", lvl_out['bbox_preds'].mean().item())
            detections = model.post_process(outputs, 
                                           conf_threshold=config.CONF_THRESHOLD,
                                           nms_threshold=config.NMS_THRESHOLD)[0]
            
            # Cargar imagen original
            img_path = os.path.join(config.IMAGES_DIR, f"{data['file_name']}.png")
            image = Image.open(img_path).convert('RGB')
            
            # Crear figura
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(image)
            
            # Escalar boxes a tamaño original
            scale_x = data['original_size'][0] / config.IMAGE_SIZE
            scale_y = data['original_size'][1] / config.IMAGE_SIZE
            
            # Dibujar GT boxes en verde
            for box, label in zip(data['boxes'], data['labels']):
                x1, y1, x2, y2 = box
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='green', 
                                       facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1-5, f'GT: {config.CLASSES[label]}', 
                       color='green', fontsize=12, weight='bold')
            
            # Dibujar predicciones en rojo
            pred_count = 0
            for box, score, label in zip(detections['boxes'], 
                                        detections['scores'], 
                                        detections['labels']):
                x1, y1, x2, y2 = box
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-20, f'{config.CLASSES[label]}: {score:.2f}', 
                       color='red', fontsize=12, weight='bold')
                pred_count += 1
            
            ax.set_xlim(0, data['original_size'][0])
            ax.set_ylim(data['original_size'][1], 0)
            ax.axis('off')
            
            # Determinar si es buen o mal ejemplo
            # (simplificado: si detectó al menos un objeto es "bueno")
            if pred_count > 0 and len(data['boxes']) > 0:
                good_examples.append((fig, data['file_name']))
            else:
                bad_examples.append((fig, data['file_name']))
            
            plt.close()
    
    # Guardar ejemplos
    for i, (fig, name) in enumerate(good_examples[:5]):
        fig.savefig(f'visualizations/good_example_{i}_{name}.png', 
                   bbox_inches='tight', dpi=150)
    
    for i, (fig, name) in enumerate(bad_examples[:5]):
        fig.savefig(f'visualizations/bad_example_{i}_{name}.png', 
                   bbox_inches='tight', dpi=150)
    
    print(f"Visualizaciones guardadas en 'visualizations/'")

def evaluate_with_centerness(config_path, checkpoint_path, use_centerness):
    """Evalúa el modelo con o sin centerness"""
    # Cargar configuración
    config = Config()
    config.USE_CENTERNESS = use_centerness
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Importar aquí para evitar importación circular
    from dataset import CatDogDetectionDataset, collate_fn
    from models.detector import CatDogDetector
    
    # Dataset
    val_dataset = CatDogDetectionDataset(config, split='val', use_precomputed_features=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Modelo
    # Importar aquí para evitar importación circular
    from models.detector import CatDogDetector
    
    model = CatDogDetector(config, use_pretrained_encoder=True).to(device)
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluar
    print(f"\nEvaluando modelo {'CON' if use_centerness else 'SIN'} centerness...")
    results = evaluate_map(model, val_loader, config, device)
    
    print(f"mAP: {results['mAP']:.4f}")
    for class_name, ap in zip(config.CLASSES, results['AP']):
        print(f"  {class_name}: {ap:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo de detección')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path al checkpoint del modelo')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualizar predicciones')
    parser.add_argument('--compare-centerness', action='store_true',
                       help='Comparar resultados con y sin centerness')
    args = parser.parse_args()
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.compare_centerness:
        # Evaluar con y sin centerness
        results_with = evaluate_with_centerness(None, args.checkpoint, True)
        results_without = evaluate_with_centerness(None, args.checkpoint, False)
        
        print("\n=== COMPARACIÓN DE RESULTADOS ===")
        print(f"CON Centerness - mAP: {results_with['mAP']:.4f}")
        print(f"SIN Centerness - mAP: {results_without['mAP']:.4f}")
        print(f"Diferencia: {results_with['mAP'] - results_without['mAP']:.4f}")
    else:
        # Importar aquí para evitar importación circular
        from dataset import CatDogDetectionDataset, collate_fn
        from models.detector import CatDogDetector
        
        # Evaluación estándar
        val_dataset = CatDogDetectionDataset(config, split='val', use_precomputed_features=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                               shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        model = CatDogDetector(config, use_pretrained_encoder=True).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        results = evaluate_map(model, val_loader, config, device)
        
        print(f"\nmAP: {results['mAP']:.4f}")
        for class_name, ap in zip(config.CLASSES, results['AP']):
            print(f"  {class_name}: {ap:.4f}")
        
        if args.visualize:
            visualize_predictions(model, val_dataset, config, device)

if __name__ == '__main__':
    main()