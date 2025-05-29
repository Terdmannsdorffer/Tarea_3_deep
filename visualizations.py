# generate_visualizations.py - Generar las visualizaciones requeridas para la tarea

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def load_model_and_config(checkpoint_path):
    """Carga modelo y configuración desde checkpoint"""
    from config import Config
    from models.detector import CatDogDetector
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Cargar configuración del checkpoint
    config = checkpoint.get('config', Config())
    
    # Crear y cargar modelo
    model = CatDogDetector(config, use_pretrained_encoder=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, device

def generate_visualizations_for_model(checkpoint_path, output_dir, experiment_name):
    """Genera visualizaciones para un modelo específico"""
    print(f"\n{'='*60}")
    print(f"Generando visualizaciones para: {experiment_name}")
    print(f"{'='*60}")
    
    # Cargar modelo
    model, config, device = load_model_and_config(checkpoint_path)
    
    # Cargar dataset de validación
    from dataset import CatDogDetectionDataset
    val_dataset = CatDogDetectionDataset(config, split='val', use_precomputed_features=True)
    
    # Crear directorio de salida
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Procesar TODAS las imágenes para encontrar buenos y malos ejemplos
    print(f"Procesando {len(val_dataset)} imágenes de validación...")
    
    all_results = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Evaluando"):
            data = val_dataset[idx]
            features = data['features'].unsqueeze(0).to(device)
            
            # Predicciones
            outputs = model(features)
            detections = model.post_process(
                outputs, 
                conf_threshold=0.3,  # Umbral más bajo para ver más detecciones
                nms_threshold=0.5
            )[0]
            
            # Calcular métricas para esta imagen
            has_gt = len(data['boxes']) > 0
            has_pred = len(detections['boxes']) > 0
            
            # Calcular IoU si hay predicciones y GT
            max_iou = 0.0
            correct_class = False
            
            if has_gt and has_pred:
                gt_box = data['boxes'][0]  # Solo hay un objeto por imagen
                gt_label = data['labels'][0]
                
                for pred_box, pred_label, score in zip(detections['boxes'], 
                                                       detections['labels'],
                                                       detections['scores']):
                    # Calcular IoU
                    iou = compute_iou(gt_box, pred_box)
                    if iou > max_iou:
                        max_iou = iou
                        correct_class = (pred_label == gt_label)
            
            # Guardar resultados
            all_results.append({
                'idx': idx,
                'file_name': data['file_name'],
                'has_gt': has_gt,
                'has_pred': has_pred,
                'max_iou': max_iou,
                'correct_class': correct_class,
                'max_score': detections['scores'].max().item() if has_pred else 0.0,
                'data': data,
                'detections': detections
            })
    
    # Ordenar por calidad (IoU alto y clase correcta = bueno)
    all_results.sort(key=lambda x: (x['max_iou'] * x['correct_class']), reverse=True)
    
    # Seleccionar ejemplos
    good_examples = [r for r in all_results if r['max_iou'] > 0.5 and r['correct_class']][:5]
    
    # Para malos ejemplos, buscar diferentes tipos de errores
    false_positives = [r for r in all_results if r['has_pred'] and not r['has_gt']][:2]
    false_negatives = [r for r in all_results if r['has_gt'] and not r['has_pred']][:2]
    wrong_class = [r for r in all_results if r['has_pred'] and r['has_gt'] and not r['correct_class']][:1]
    bad_examples = false_positives + false_negatives + wrong_class
    
    # Si no hay suficientes buenos ejemplos, tomar los mejores disponibles
    if len(good_examples) < 5:
        good_examples = all_results[:5]
    
    # Si no hay suficientes malos ejemplos, tomar los peores
    if len(bad_examples) < 5:
        bad_examples = all_results[-5:]
    
    print(f"\nEncontrados:")
    print(f"  - {len(good_examples)} buenos ejemplos (IoU > 0.5)")
    print(f"  - {len(bad_examples)} malos ejemplos")
    
    # Generar visualizaciones
    print("\nGenerando imágenes...")
    
    # Buenos ejemplos
    for i, result in enumerate(good_examples):
        fig = create_visualization(result, config)
        fig.savefig(os.path.join(vis_dir, f'good_example_{i}_{result["file_name"]}.png'),
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    # Malos ejemplos
    for i, result in enumerate(bad_examples):
        fig = create_visualization(result, config)
        fig.savefig(os.path.join(vis_dir, f'bad_example_{i}_{result["file_name"]}.png'),
                   bbox_inches='tight', dpi=150)
        plt.close(fig)
    
    # Generar gráfico de distribución de scores
    create_score_distribution_plot(all_results, vis_dir, experiment_name)
    
    # Generar resumen
    create_summary_plot(good_examples, bad_examples, vis_dir, experiment_name)
    
    print(f"\n✅ Visualizaciones guardadas en: {vis_dir}")
    
    return all_results

def compute_iou(box1, box2):
    """Calcula IoU entre dos boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersección
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Unión
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def create_visualization(result, config):
    """Crea una visualización para un resultado"""
    data = result['data']
    detections = result['detections']

    # Cargar imagen
    img_path = os.path.join(config.IMAGES_DIR, f"{data['file_name']}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(config.IMAGES_DIR, f"{data['file_name']}.jpg")

    image = Image.open(img_path).convert('RGB')

    # Crear figura
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Escalar boxes
    scale_x = data['original_size'][0] / config.IMAGE_SIZE
    scale_y = data['original_size'][1] / config.IMAGE_SIZE

    # Dibujar GT en verde
    for box, label in zip(data['boxes'], data['labels']):
        device = box.device
        scales = torch.tensor([scale_x, scale_y, scale_x, scale_y], device=device)
        x1, y1, x2, y2 = (box * scales).cpu()
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=3, edgecolor='green',
                                 facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f'GT: {config.CLASSES[label]}',
                color='white', fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.8))

    # Dibujar predicciones en rojo
    for box, score, label in zip(detections['boxes'],
                                 detections['scores'],
                                 detections['labels']):
        device = box.device
        scales = torch.tensor([scale_x, scale_y, scale_x, scale_y], device=device)
        x1, y1, x2, y2 = (box * scales).cpu()
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=3, edgecolor='red',
                                 facecolor='none')
        ax.add_patch(rect)
        ax.text(x2, y2 + 10, f'{config.CLASSES[label]}: {score:.2f}',
                color='white', fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8))

    # Título con información
    title = f'{data["file_name"]} - '
    if result['max_iou'] > 0.5 and result['correct_class']:
        title += f'BUENO (IoU: {result["max_iou"]:.2f})'
    else:
        if not result['has_pred']:
            title += 'MALO (No detectado)'
        elif not result['has_gt']:
            title += 'MALO (Falso positivo)'
        elif not result['correct_class']:
            title += f'MALO (Clase incorrecta, IoU: {result["max_iou"]:.2f})'
        else:
            title += f'MALO (IoU bajo: {result["max_iou"]:.2f})'

    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlim(0, data['original_size'][0])
    ax.set_ylim(data['original_size'][1], 0)
    ax.axis('off')

    return fig


def create_score_distribution_plot(all_results, output_dir, experiment_name):
    """Crea gráfico de distribución de scores"""
    scores = [r['max_score'] for r in all_results if r['has_pred']]
    
    if len(scores) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Score de Confianza', fontsize=12)
        plt.ylabel('Número de Detecciones', fontsize=12)
        plt.title(f'Distribución de Scores - {experiment_name}', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'score_distribution.png'), 
                    bbox_inches='tight', dpi=150)
        plt.close()

def create_summary_plot(good_examples, bad_examples, output_dir, experiment_name):
    """Crea un resumen visual de los resultados"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Estadísticas
    categories = ['Buenos\n(IoU>0.5)', 'Malos\n(IoU<0.5)']
    counts = [len(good_examples), len(bad_examples)]
    colors = ['green', 'red']
    
    ax1.bar(categories, counts, color=colors, alpha=0.7)
    ax1.set_ylabel('Número de Ejemplos', fontsize=12)
    ax1.set_title('Distribución de Calidad', fontsize=14, weight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Pie chart de tipos de errores
    error_types = {'No detectado': 0, 'Falso positivo': 0, 'Clase incorrecta': 0, 'IoU bajo': 0}
    
    for r in bad_examples:
        if not r['has_pred']:
            error_types['No detectado'] += 1
        elif not r['has_gt']:
            error_types['Falso positivo'] += 1
        elif not r['correct_class']:
            error_types['Clase incorrecta'] += 1
        else:
            error_types['IoU bajo'] += 1
    
    # Filtrar tipos con 0 errores
    error_types = {k: v for k, v in error_types.items() if v > 0}
    
    if error_types:
        ax2.pie(error_types.values(), labels=error_types.keys(), autopct='%1.1f%%',
                colors=['orange', 'red', 'purple', 'brown'])
        ax2.set_title('Tipos de Errores', fontsize=14, weight='bold')
    
    plt.suptitle(f'Resumen de Resultados - {experiment_name}', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary.png'), bbox_inches='tight', dpi=150)
    plt.close()

def main():
    """Genera visualizaciones para ambos experimentos"""
    experiments = [
        ('experiments/with_centerness/checkpoints/best_model.pth', 
         'experiments/with_centerness', 'CON Centerness'),
        ('experiments/without_centerness/checkpoints/best_model.pth', 
         'experiments/without_centerness', 'SIN Centerness')
    ]
    
    for checkpoint_path, output_dir, name in experiments:
        if os.path.exists(checkpoint_path):
            generate_visualizations_for_model(checkpoint_path, output_dir, name)
        else:
            print(f"\n⚠️  No se encontró checkpoint: {checkpoint_path}")
            print(f"   Asegúrate de haber entrenado el modelo {name}")
    
    print("\n✅ Todas las visualizaciones generadas!")
    print("📁 Revisa las carpetas:")
    print("   - experiments/with_centerness/visualizations/")
    print("   - experiments/without_centerness/visualizations/")

if __name__ == '__main__':
    main()