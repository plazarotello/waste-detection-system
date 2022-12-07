import torch
from lightning import Trainer
from lightning.lite.utilities.seed import seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint
)
from lightning.pytorch.loggers import NeptuneLogger


from pandas import DataFrame
from pathlib import Path
from typing import Tuple, Union
from sklearn.model_selection import train_test_split


import importlib_metadata
from neptune.new.types import File
import pathlib


from . import shared_data as base
from . import waste_detection_module
from .waste_detection_module import WasteDetectionModule



def split_dataset(dataset : DataFrame) -> Tuple[DataFrame, DataFrame]:
    paths = dataset.path.unique()
    train_paths, val_paths = train_test_split(paths, test_size=0.15)

    return (dataset[dataset['path'].isin(train_paths)],
           dataset[dataset['path'].isin(val_paths)])



def tune(model, train_dataset):
    seed_everything(base.SEED)

    epochs = 100

    if base.USE_GPU:
        trainer = Trainer(
            gpus=base.GPU,
            num_sanity_val_steps=0,
            max_epochs=epochs,
            accelerator='gpu', devices=1,
            auto_lr_find=True,
            auto_scale_batch_size=True
        )
    else:
        trainer = Trainer(
            accelerator='cpu',
            num_sanity_val_steps=0,
            max_epochs=epochs,
            auto_lr_find=True,
            auto_scale_batch_size=True
        )

    lit_model = WasteDetectionModule(model=model, train_dataset=train_dataset,
        val_dataset=None, batch_size=1, lr=0.1)

    print('Searching for optimal initial learning rate...')
    lr_finder = trainer.tuner.lr_find(model=lit_model, min_lr=1e-9)
    
    if lr_finder: 
        print(f'Suggested initial lr: {lr_finder.suggestion()}')
        print(f'Complete results: {lr_finder.results}')
        lr_finder.plot(suggest=True, show=True)

    
    print('Searching for maximum batch size...')
    batch_scaler = trainer.tuner.scale_batch_size(model=lit_model, init_val=1)
    print(f'Largest batch_size allowed: {batch_scaler}')


def train(model, train_dataset : DataFrame, val_dataset : DataFrame, config : dict, 
        neptune_project : str, limit_validation : Union[bool, float] = False):
    seed_everything(base.SEED)

    epochs = config['epochs']
    lr = config['lr']
    bs = config['bs']
    output_dir = config['checkpoint_dir']

    if type(limit_validation) is bool:
        if limit_validation: limit_validation = base.LIMIT_VAL_BATCHES
        else: limit_validation = 1.0

    neptune_logger = NeptuneLogger(
        api_key=base.NEPTUNE_API_KEY,
        project=neptune_project
    )

    lit_model = WasteDetectionModule(model=model, train_dataset=train_dataset,
        val_dataset=val_dataset, batch_size=bs, lr=lr)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='Validation_mAP', 
        mode='max',
        save_top_k=3,
        filename='ssd-{epoch:02d}-{Validation_mAP:.2f}',
        verbose=False,
        save_last=True
        )
    learningrate_callback = LearningRateMonitor(
        logging_interval='step',
        log_momentum=False
    )
    earlystopping_callback = EarlyStopping(
        monitor='Validation_mAP',
        patience=base.PATIENCE,
        mode='max'
    )

    if base.USE_GPU:
        trainer = Trainer(
            gpus=base.GPU,
            callbacks=[checkpoint_callback, learningrate_callback,
                earlystopping_callback],
            default_root_dir=Path(output_dir),
            max_epochs=epochs,
            logger=neptune_logger,
            accelerator='gpu',
            num_sanity_val_steps=0,
            auto_lr_find=False,
            auto_scale_batch_size=False,
            limit_val_batches=limit_validation
        )
    else:
        trainer = Trainer(
            gpus=base.GPU,
            callbacks=[checkpoint_callback, learningrate_callback,
                earlystopping_callback],
            default_root_dir=Path(output_dir),
            max_epochs=epochs,
            logger=neptune_logger,
            num_sanity_val_steps=0,
            auto_lr_find=False,
            auto_scale_batch_size=False,
            limit_val_batches=limit_validation
        )

    trainer.fit(model=lit_model)
    
    log_packages_neptune(neptune_logger=neptune_logger)
    
    best_k_models = checkpoint_callback.best_k_models
    last_model_path = Path(checkpoint_callback.last_model_path)
    best_model_path = None
    if best_k_models:
        for path, score in best_k_models.items():
            path = Path(path)
            if path.exists() and path.is_file():
                if not best_model_path:
                    best_model_path = path

                model_name = path.stem + str(score.double()) + path.suffix
                log_model_neptune(
                    checkpoint_path=path, 
                    save_directory=Path(output_dir),
                    neptune_logger=neptune_logger,
                    name=model_name
                )
    elif last_model_path.exists() and last_model_path.is_file():
        best_model_path = last_model_path
        log_model_neptune(
            checkpoint_path=best_model_path, 
            save_directory=Path(output_dir),
            neptune_logger=neptune_logger,
            name=last_model_path.name
        )
                
    neptune_logger.experiment.stop()
    
    return best_model_path, trainer


def test(trainer : Trainer, dataset : DataFrame):
    dataloader = waste_detection_module.get_dataloader(
        data=dataset,
        batch_size=1,
        shuffle=False
        )
    
    return trainer.test(ckpt_path='best', dataloaders=dataloader)


def log_packages_neptune(neptune_logger):
    """log the packages of the current python env."""
    dists = importlib_metadata.distributions()
    packages = {
        idx: (dist.metadata["Name"], dist.version) for idx, dist in enumerate(dists)
    }

    packages_df = DataFrame.from_dict(
        packages, orient="index", columns=["package", "version"]
    )

    neptune_logger.experiment['packages'].upload(File.as_html(packages_df))


def log_model_neptune(
    checkpoint_path: pathlib.Path,
    save_directory: pathlib.Path,
    name: str,
    neptune_logger,
):
    """Saves the model to disk, uploads it to neptune."""
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["hyper_parameters"]["model"]
    torch.save(model.state_dict(), save_directory / name)

    neptune_logger['model'].upload(File.as_pickle(model))