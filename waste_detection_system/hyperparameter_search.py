from pathlib import Path
import torch
import pandas as pd
from codecarbon import EmissionsTracker

from . import models, trainer, shared_data as base


def hyperparameter_search(name: str, config: dict, dataset : pd.DataFrame,
        selected_model : models.AVAILABLE_MODELS, num_classes : int, 
        tll : int, weights = None):

    base_model = models.get_base_model(num_classes, selected_model, tll)
    assert base_model is not None
    if weights:
        base_model = models.load_partial_weights(base_model, weights)

    model = base_model

    gpu_ids = [base.GPU] if torch.cuda.is_available() and base.USE_GPU else None
    tracker = EmissionsTracker(project_name=name, experiment_id=f'hypersearch-{name}', 
        gpu_ids=gpu_ids, log_level='error', tracking_mode='process', 
        measure_power_secs=30, output_file=Path(config['results_dir']) / f'{name}-emissions.csv')  # type: ignore
    

    tracker.start()
    trainer.tune(model=model, train_dataset=dataset)
    tracker.stop()        