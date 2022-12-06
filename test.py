from pathlib import Path
import os
# -------------------------------------------------------------
import pandas as pd
# -------------------------------------------------------------
import seaborn as sns
sns.set()
# -------------------------------------------------------------
import torch
# -------------------------------------------------------------
# MSW Detector packages
from waste_detection_system import shared_data as base, main

# =============================================================
os.environ["PYTHONUNBUFFERED"] = '1'



with open(base.FINAL_DATA_CSV, 'r', encoding='utf-8-sig') as final_file:
  final_dataset = pd.read_csv(final_file)

zerowaste = final_dataset[final_dataset['dataset'] == 'final']
zerowaste_train = zerowaste[zerowaste['type'] == 'train']
zerowaste_val = zerowaste[zerowaste['type'] == 'val']
zerowaste_test = zerowaste[zerowaste['type'] == 'test']

resortit = final_dataset[final_dataset['dataset'] == 'complementary']
resortit_train = resortit[resortit['type'] == 'train']
resortit_val = resortit[resortit['type'] == 'val']
resortit_test = resortit[resortit['type'] == 'test']

# ------------------------------------------------------------------------
zerowaste_train_sample = zerowaste_train[zerowaste_train['path'].isin( 
  zerowaste_train[['path']].sample(frac=0.3).path.tolist())]

resortit_train_sample = resortit_train[resortit_train['path'].isin(
    resortit_train[['path']].sample(frac=0.3).path.tolist())]

main.train(train_dataset=resortit_train, val_dataset=resortit_val, name='ssd-resortit', 
            config=base.MODELS_DIR/'ssd-pretrain.json', num_classes=2, tll=1, resortit_zw=0,
            selected_model=main.models.AVAILABLE_MODELS.SSD)