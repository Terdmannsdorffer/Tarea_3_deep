import os

class Config:
    # Paths
    DATA_ROOT = r"C:\Users\Usuario\Desktop\deep\Tarea_3\cat_dog_det"
    IMAGES_DIR = os.path.join(DATA_ROOT, "images")
    ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "annotations")
    TRAIN_FILE = os.path.join(DATA_ROOT, "train.txt")
    VAL_FILE = os.path.join(DATA_ROOT, "val.txt")
    FEATURES_DIR = os.path.join(DATA_ROOT, "dino_features")
    
    # Model parameters
    NUM_CLASSES = 2
    IMAGE_SIZE = 224
    PATCH_SIZE = 14
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    FEATURE_DIM = 384
    
    # FCOS parameters
    FPN_CHANNELS = 256
    STRIDES = [8, 16, 32, 64, 128]
    USE_CENTERNESS = True
    
    # Training parameters - OPTIMIZADO PARA RTX 3090
    BATCH_SIZE = 64  # Aumentado significativamente
    LEARNING_RATE = 2e-3  # Ajustado para batch size mayor
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-4
    
    # DataLoader optimizations
    NUM_WORKERS = 8
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    
    # Mixed Precision
    USE_AMP = True
    
    # PyTorch optimizations
    COMPILE_MODEL = True
    CUDNN_BENCHMARK = True
    
    # Loss weights
    CLS_LOSS_WEIGHT = 1.0
    REG_LOSS_WEIGHT = 1.0
    CENTERNESS_LOSS_WEIGHT = 1.0
    
    # Detection parameters
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.5
    
    # Classes
    CLASSES = ["dog", "cat"]
    CLASS_TO_IDX = {"dog": 0, "cat": 1}