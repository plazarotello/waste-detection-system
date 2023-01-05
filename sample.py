# -*- coding: utf-8 -*-

"""
Sample script illustrating the usage of the main functions of
the module. In order of apperarance:

- Load the datasets
	- Sample the datasets to make the script more lightweighted
- Creates and prints the structure and trainable layers of some
	models
- Searches the optimal learning rate with the lightning tuner
- Trains a model
- Loads a model checkpoint
- Tests the model checkpoint against the test dataset (sample)
- Benchmarks the inference time against the test dataset (sample)
"""

# -------------------------------------------------------------
import os
import pandas as pd
# -------------------------------------------------------------
# MSW Detector packages
from waste_detection_system import shared_data as base, main
# =============================================================
os.environ["PYTHONUNBUFFERED"] = '1'
# =============================================================


with open(base.FINAL_DATA_CSV, 'r', encoding='utf-8-sig') as final_file:
	final_dataset = pd.read_csv(final_file)

# ------------------------------------------------------------------------
zerowaste = final_dataset[final_dataset['dataset'] == 'final']
zerowaste_train = zerowaste[zerowaste['type'] == 'train']
zerowaste_val = zerowaste[zerowaste['type'] == 'val']
zerowaste_test = zerowaste[zerowaste['type'] == 'test']

zerowaste_train_sample = zerowaste_train[zerowaste_train['path'].isin( 
	zerowaste_train[['path']].sample(frac=0.3).path.tolist())]
zerowaste_val_sample = zerowaste_val[zerowaste_val['path'].isin( 
	zerowaste_val[['path']].sample(frac=0.3).path.tolist())]
zerowaste_test_sample = zerowaste_test[zerowaste_test['path'].isin( 
	zerowaste_test[['path']].sample(frac=0.3).path.tolist())]

# ------------------------------------------------------------------------
resortit = final_dataset[final_dataset['dataset'] == 'complementary']
resortit_train = resortit[resortit['type'] == 'train']
resortit_val = resortit[resortit['type'] == 'val']
resortit_test = resortit[resortit['type'] == 'test']

resortit_train_sample = resortit_train[resortit_train['path'].isin(
    resortit_train[['path']].sample(frac=0.3).path.tolist())]
resortit_val_sample = resortit_val[resortit_val['path'].isin(
    resortit_val[['path']].sample(frac=0.3).path.tolist())]

# ------------------------------------------------------------------------
print('FASTER R-CNN / TLL=3')
main.models.pretty_summary(main.models.get_fasterrcnn(2, 3))
print('SSD /TLL=1')
main.models.pretty_summary(main.models.get_ssd(2, 1))

# ------------------------------------------------------------------------
print('HYPERPARAMETER SEARCH')
main.hyperparameter_search(
	name='ssd-zerowaste-hyper', 
	config=base.MODELS_DIR/'hyper-options.json', 
    dataset=zerowaste_train_sample, 
	selected_model=main.models.AVAILABLE_MODELS.SSD, 
	num_classes=2,
	tll=3, 
	weights=None,
	find_batch_size = False, 
	metric='Validation_mAP'
	)

# ------------------------------------------------------------------------
print('TRAIN')
main.train(
	train_dataset=resortit_train_sample, 
	val_dataset=resortit_val_sample, 
	name='fasterrcnn-resortit-sample', 
    config=base.MODELS_DIR/'faster-r-cnn-sample.json', 
	num_classes=2, 
	tll=1, 
	resortit_zw=0,
	selected_model=main.models.AVAILABLE_MODELS.SSD, 
	limit_validation=0.1, 
	metric='training_loss'
	)

# ------------------------------------------------------------------------
best_model_path_tll0 = base.MODELS_DIR / 'fasterrcnn-zerowaste' / 'tll0_3.ckpt'

# ------------------------------------------------------------------------
print('TEST')
main.test(
	best_model_path_tll0, 
	selected_model=main.models.AVAILABLE_MODELS.FASTERRCNN, 
	resortit_zw=1, 
	test_dataset=zerowaste_test_sample
	)

# ------------------------------------------------------------------------
print('BENCHMARK')
average_ms = main.benchmark(
	best_model_path_tll0, 
	test_dataset=zerowaste_test_sample
	)