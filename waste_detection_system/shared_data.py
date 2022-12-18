# -*- coding: utf-8 -*-

"""
This script holds different constants and enumerators needed in the
rest of the scripts in the module. It defines the directories's structure 
as such:

::

    root
    ├── candidate-dataset: temporary folder to hold a dataset while EDA
    ├── complementary-dataset: ResortIT dataset ready for training
    ├── config
    │   ├── models: configurations for the models
    │   └── config.json: general configuration
    ├── dataset: ZeroWaste dataset ready for training
    ├── raw-datasets
    │   ├── cig_butts
    │   ├── CompostNet
    │   ├── drinking-waste
    │   ├── ResortIt
    │   ├── TACO
    │   ├── Trashbox-metal
    │   ├── WasteClassification
    │   └── zero-waste
    │       └── zerowaste-f
    ├── results: holds the results
    │   └── final_dataset.csv
    └── *waste_detection_system*: the source code

"""

from enum import Enum
from shutil import rmtree
from pathlib import Path
import os
import json

SEED = 11
"""int: global seed
"""
PATIENCE = 30
"""int: epochs of patience before considering no progress is being made
"""

IMG_WIDTH = 640
"""int: pixels of width for resizing images
"""
IMG_HEIGHT = 640
"""int: pixels of height for resizing images
"""

AVAILABLE_MODELS = Enum('Models', 'FASTERRCNN FCOS RETINANET SSD')
"""enum: available models: ``FasterRCNN``, ``FCOS``, ``RetinaNet``, ``SSD``
"""

AVAILABLE_CLASSIFIERS = Enum('Classifiers', 'SVM KNN')
"""enum available classifiers: ``SVM``, ``KNN``
"""

ROOT = Path(os.path.dirname(__file__)).parent

RESULTS = ROOT / Path('results')

TENSORBOARD_DIR = ROOT /Path('tensorboard_logs')


def get_project_name(model : AVAILABLE_MODELS, resortit_zw : int) -> str:
    model_to_project = {
        AVAILABLE_MODELS.FASTERRCNN : 'faster-r-cnn',
        AVAILABLE_MODELS.SSD : 'ssd',
        AVAILABLE_MODELS.FCOS : 'fcos',
        AVAILABLE_MODELS.RETINANET : 'retinanet'
    }
    dataset_to_project = {
        0 : 'resortit',
        1 : 'zerowaste'
    }

    return f'{model_to_project[model]}_{dataset_to_project[resortit_zw]}'

def get_experiment_name(tll : int) -> str:
    return f'tll_{tll}'



# -----------------------------------------------------------------------------
# Raw datasets paths
# =============================================================================

RAW_ROOT = ROOT / Path('raw-datasets')

CANDIDATE_DATASET = ROOT / Path('candidate-dataset')
COMP_DATASET = ROOT / Path('complementary-dataset')
FINAL_DATASET = ROOT / Path('dataset')


CIG_BUTTS = RAW_ROOT / 'cig_butts'
DRINKING_WASTE = RAW_ROOT / 'drinking-waste'
TACO = RAW_ROOT / 'TACO'
ZERO_WASTE = RAW_ROOT / 'zero-waste' / 'zerowaste-f'
TRASHBOX_METAL = RAW_ROOT / 'Trashbox-metal'
WASTE_CL = RAW_ROOT / 'WasteClassification'
COMPOSTNET = RAW_ROOT / 'CompostNet'
RESORTIT = RAW_ROOT / 'ResortIt'


CIG_BUTTS_CSV = RAW_ROOT / 'cig-butts.csv'
DRINKING_WASTE_CSV = RAW_ROOT / 'drinking-waste.csv'
TACO_CSV = RAW_ROOT / 'TACO.csv'
ZERO_WASTE_CSV = RAW_ROOT / 'zerowaste.csv'
TRASHBOX_METAL_CSV = RAW_ROOT / 'trashbox-metal.csv'
WASTE_CL_CSV = RAW_ROOT / 'waste-cl.csv'
COMPOSTNET_CSV = RAW_ROOT / 'compostnet.csv'
RESORTIT_CSV = RAW_ROOT / 'resortit.csv'


# create directories
RAW_ROOT.mkdir(parents=True, exist_ok=True)
rmtree(CANDIDATE_DATASET, ignore_errors=True)
CANDIDATE_DATASET.mkdir(parents=True, exist_ok=True)
COMP_DATASET.mkdir(parents=True, exist_ok=True)
FINAL_DATASET.mkdir(parents=True, exist_ok=True)
CIG_BUTTS.mkdir(parents=True, exist_ok=True)
DRINKING_WASTE.mkdir(parents=True, exist_ok=True)
TACO.mkdir(parents=True, exist_ok=True)
ZERO_WASTE.mkdir(parents=True, exist_ok=True)
TRASHBOX_METAL.mkdir(parents=True, exist_ok=True)
WASTE_CL.mkdir(parents=True, exist_ok=True)
COMPOSTNET.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Model's config paths
# =============================================================================

CONFIG_DIR = ROOT / Path('config')

MODELS_DIR = CONFIG_DIR / Path('models')
PSEUDOLABELLING_DIR = CONFIG_DIR / Path('pseudo-labelling')


# create directories
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)



LIMIT_VAL_BATCHES = 0.25
"""float: by default, limit to 25% of the validation dataset
"""

USE_CPU = None
"""Union[bool, None]: if the model training must be done on CPU
"""
USE_GPU = None
"""Union[bool, None]: if the model training must be done on GPU
"""
GPU = None
"""Union[int, None]: the GPU identifier the model training must be done
"""


def _base_configuration():
    config_path = CONFIG_DIR / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        global USE_CPU, USE_GPU, GPU
        USE_CPU = config['use_cpu']
        USE_GPU = config['use_gpu']
        GPU = config['gpu']

_base_configuration()


# =============================================================================
# Data paths
# =============================================================================

FINAL_DATA_CSV = RESULTS / 'final-dataset.csv'
COMP_DATA_CSV = RESULTS / 'complementary-dataset.csv'

# =============================================================================
# Color constants
# =============================================================================

COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (229, 190, 1)
COLOR_BROWN = (128, 64, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_GREY = (128, 128, 128)

# =============================================================================
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
    DRINKING_WASTE : 'drinkingwaste',
    RESORTIT : 'resortit'
}

# =============================================================================
# Raw to cooked dataset label conversion
# =============================================================================

RELATION_CATS = {
    'cardboard'.upper(): CATS_PAPEL,
    'soft_plastic'.upper(): CATS_PLASTICO,
    'rigid_plastic'.upper(): CATS_PLASTICO,
    'metal'.upper(): CATS_METAL,

    'Alu'.upper(): CATS_METAL,
    'Carton'.upper(): CATS_PAPEL,
    'Nylon'.upper(): CATS_PLASTICO,
    'Bottle'.upper(): CATS_PLASTICO,

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
