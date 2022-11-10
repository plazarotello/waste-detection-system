
from pathlib import Path
import pandas as pd

from waste_detection_system import shared_data as base, main


with open(base.TRASHBOX_METAL_CSV, 'r') as f:
    metals = pd.read_csv(f)

main.hyperparameter_search(metals, 'trashbox-metal-hyper', Path('config')/'models'/'hyper-opt.json', main.models.AVAILABLE_MODELS.FASTERRCNN,2)