# -*- coding: utf-8 -*-

from pathlib import Path

SEED = 11

IMG_WIDTH = 640
IMG_HEIGHT = 640

ROOT = Path('../..')
RESULTS = ROOT / Path('results')

# -----------------------------------------------------------------------------
# Raw datasets paths
# =============================================================================

RAW_ROOT = ROOT / Path('raw-datasets')
MOD_DATASET = ROOT / Path('dataset')

CIG_BUTTS = RAW_ROOT / 'cig_butts'
DRINKING_WASTE = RAW_ROOT / 'drinking-waste'
TACO = RAW_ROOT / 'TACO'
ZERO_WASTE = RAW_ROOT / 'zero-waste' / 'zerowaste-f'
TRASHBOX_METAL = RAW_ROOT / 'Trashbox-metal'
WASTE_CL = RAW_ROOT / 'WasteClassification'
COMPOSTNET = RAW_ROOT / 'CompostNet'

CIG_BUTTS_CSV = RAW_ROOT / 'cig-butts.csv'
DRINKING_WASTE_CSV = RAW_ROOT / 'drinking-waste.csv'
TACO_CSV = RAW_ROOT / 'TACO.csv'
ZERO_WASTE_CSV = RAW_ROOT / 'zerowaste.csv'
TRASHBOX_METAL_CSV = RAW_ROOT / 'trashbox-metal.csv'
WASTE_CL_CSV = RAW_ROOT / 'waste-cl.csv'
COMPOSTNET_CSV = RAW_ROOT / 'compostnet.csv'


RAW_ROOT.mkdir(parents=True, exist_ok=True)
MOD_DATASET.mkdir(parents=True, exist_ok=True)
CIG_BUTTS.mkdir(parents=True, exist_ok=True)
DRINKING_WASTE.mkdir(parents=True, exist_ok=True)
TACO.mkdir(parents=True, exist_ok=True)
ZERO_WASTE.mkdir(parents=True, exist_ok=True)
TRASHBOX_METAL.mkdir(parents=True, exist_ok=True)
WASTE_CL.mkdir(parents=True, exist_ok=True)
COMPOSTNET.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Model's config paths
# =============================================================================

MODELS_DIR = ROOT / Path('config') / Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------

YOLO_MODEL = MODELS_DIR / 'ultralytics_yolov5_master'
YOLO_BEST_WEIGHTS = ROOT / 'yolov5l6-best.pt'
YOLO_HIDDEN_WEIGHTS = YOLO_MODEL / 'runs' / \
    'train' / 'exp' / 'weights' / 'best.pt'
YOLO_HIDDEN_HYPER = YOLO_MODEL / 'runs' / \
    'train' / 'exp' / 'hyp.yaml'
YOLO_BEST_HYPER = ROOT / 'yolov5-hyper.yaml'

# -----------------------------------------------------------------------------
# Data paths
# =============================================================================

DATA_CSV = RESULTS / 'dataset.csv'

YOLO_CONFIG = MODELS_DIR / 'YOLO-cfg'

YOLO_NET_YAML = YOLO_CONFIG / 'custom-yolov5l.yaml'
YOLO_DATA_YAML = YOLO_CONFIG / 'dataset.yaml'
YOLO_BASE_WEIGHTS = ROOT / 'yolov5l6.pt'

YOLO_DATA_FOLDER = ROOT / 'models' / 'dataset' / 'yolo'

YOLO_DATA_IMG_TXT = 'images.txt'

YOLO_DATA_IMGS = 'images'
YOLO_DATA_LABELS = 'labels'

YOLO_DATA_TRAIN = 'train'
YOLO_DATA_VAL = 'val'
YOLO_DATA_TEST = 'test'

# -----------------------------------------------------------------------------
# Training arguments
# =============================================================================

YOLO_IMG_SIZE_W = IMG_WIDTH
YOLO_BATCH_SIZE = -1    # autobatch
YOLO_OPTIMIZER = 'SGD'
YOLO_EPOCHS = 300
YOLO_PATIENCE = 25  # epochs for Early Stopping

# Hyperparameter search
YOLO_EVOLVE_GENS = 50 # generations to search and mutate
YOLO_EVOLVE_EPOCHS = 10 # epochs to train each generation

# -----------------------------------------------------------------------------
# Color constants
# =============================================================================

COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (229, 190, 1)
COLOR_BROWN = (128, 64, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_GREY = (128, 128, 128)

# -----------------------------------------------------------------------------
# Recycling categories: categories translations to indexes and colors
# =============================================================================

CATS_PAPEL = 'PAPEL'
CATS_PLASTICO = 'PLASTICO'
CATS_METAL = 'METAL'
CATS_ORGANICO = 'ORGANICO'
CATS_VIDRIO = 'VIDRIO'
CATS_OTROS = 'OTROS'

# -----------------------------------------------------------------------------

COLOR_CATS = {
    CATS_PAPEL: COLOR_BLUE,
    CATS_PLASTICO: COLOR_YELLOW,
    CATS_METAL: COLOR_GREY,
    CATS_ORGANICO: COLOR_BROWN,
    CATS_VIDRIO: COLOR_GREEN,
    CATS_OTROS: COLOR_GREY
}

# -----------------------------------------------------------------------------

IDX_CATS = {
    CATS_PAPEL: 0,
    CATS_PLASTICO: 1,
    CATS_METAL: 2,
    CATS_ORGANICO: 3,
    CATS_VIDRIO: 4,
    CATS_OTROS: 5,
}

CATS_IDX = {
    0: CATS_PAPEL,
    1: CATS_PLASTICO,
    2: CATS_METAL,
    3: CATS_ORGANICO,
    4: CATS_VIDRIO,
    5: CATS_OTROS,
}

# -----------------------------------------------------------------------------

PREFIXES_CATS = {
    CIG_BUTTS : 'cigbutts', 
    WASTE_CL : 'wastecl', 
    COMPOSTNET : 'compostnet', 
    ZERO_WASTE : 'zerowaste', 
    TACO : 'taco', 
    TRASHBOX_METAL : 'trashbox', 
    DRINKING_WASTE : 'drinkingwaste'
}

# -----------------------------------------------------------------------------
# Raw to cooked dataset label conversion
# =============================================================================

RELATION_CATS = {
    'cardboard'.upper(): CATS_PAPEL,
    'soft_plastic'.upper(): CATS_PLASTICO,
    'rigid_plastic'.upper(): CATS_PLASTICO,
    'metal'.upper(): CATS_METAL,

    'Aluminium foil'.upper(): CATS_PLASTICO,
    'Battery'.upper(): CATS_OTROS,
    'Aluminium blister pack'.upper(): CATS_OTROS,
    'Carded blister pack'.upper(): CATS_OTROS,
    'Clear plastic bottle'.upper(): CATS_PLASTICO,
    'Glass bottle'.upper(): CATS_VIDRIO,
    'Other plastic bottle'.upper(): CATS_PLASTICO,
    'Plastic bottle cap'.upper(): CATS_PLASTICO,
    'Metal bottle cap'.upper(): CATS_PLASTICO,
    'Broken glass'.upper(): CATS_VIDRIO,
    'Aerosol'.upper(): CATS_PLASTICO,
    'Drink can'.upper(): CATS_PLASTICO,
    'Food can'.upper(): CATS_PLASTICO,
    'Corrugated carton'.upper(): CATS_PAPEL,
    'Drink carton'.upper(): CATS_PLASTICO,
    'Egg carton'.upper(): CATS_PAPEL,
    'Meal carton'.upper(): CATS_OTROS,
    'Pizza box'.upper(): CATS_OTROS,
    'Toilet tube'.upper(): CATS_PAPEL,
    'Other carton'.upper(): CATS_PAPEL,
    'Cigarette'.upper(): CATS_OTROS,
    'Paper cup'.upper(): CATS_OTROS,
    'Disposable plastic cup'.upper(): CATS_PLASTICO,
    'Foam cup'.upper(): CATS_OTROS,
    'Glass cup'.upper(): CATS_OTROS,
    'Other plastic cup'.upper(): CATS_PLASTICO,
    'Food waste'.upper(): CATS_ORGANICO,
    'Glass jar'.upper(): CATS_VIDRIO,
    'Plastic lid'.upper(): CATS_PLASTICO,
    'Metal lid'.upper(): CATS_PLASTICO,
    'Normal paper'.upper(): CATS_PAPEL,
    'Tissues'.upper(): CATS_OTROS,
    'Wrapping paper'.upper(): CATS_OTROS,
    'Magazine paper'.upper(): CATS_OTROS,
    'Paper bag'.upper(): CATS_PAPEL,
    'Plastified paper bag'.upper(): CATS_OTROS,
    'Garbage bag'.upper(): CATS_OTROS,
    'Single-use carrier bag'.upper(): CATS_PLASTICO,
    'Polypropylene bag'.upper(): CATS_OTROS,
    'Plastic Film'.upper(): CATS_OTROS,
    'Six pack rings'.upper(): CATS_OTROS,
    'Crisp packet'.upper(): CATS_OTROS,
    'Other plastic wrapper'.upper(): CATS_OTROS,
    'Spread tub'.upper(): CATS_PLASTICO,
    'Tupperware'.upper(): CATS_PLASTICO,
    'Disposable food container'.upper(): CATS_PLASTICO,
    'Foam food container'.upper(): CATS_OTROS,
    'Other plastic container'.upper(): CATS_PLASTICO,
    'Plastic glooves'.upper(): CATS_PLASTICO,
    'Plastic utensils'.upper(): CATS_PLASTICO,
    'Pop tab'.upper(): CATS_PLASTICO,
    'Rope'.upper(): CATS_OTROS,
    'Rope & Strings'.upper(): CATS_OTROS,
    'Scrap metal'.upper(): CATS_PLASTICO,
    'Shoe'.upper(): CATS_OTROS,
    'Squeezable tube'.upper(): CATS_OTROS,
    'Plastic straw'.upper(): CATS_PLASTICO,
    'Paper straw'.upper(): CATS_OTROS,
    'Styrofoam piece'.upper(): CATS_OTROS,
    'Other plastic'.upper(): CATS_PLASTICO,
    'Unlabeled litter'.upper(): CATS_OTROS,


    '0': CATS_PLASTICO,
    '1': CATS_VIDRIO,
    '2': CATS_PLASTICO,
    '3': CATS_PLASTICO,

    'cig_butt'.upper(): CATS_OTROS,
}

# -----------------------------------------------------------------------------
