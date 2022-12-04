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
from typing import Tuple
from sklearn.model_selection import train_test_split


import importlib_metadata
import neptune.new as neptune
from neptune.new.types import File
import pathlib


from . import shared_data as base
from . import waste_detection_dataset
from .model_module import ModelModule



def split_dataset(dataset : DataFrame) -> Tuple[DataFrame, DataFrame]:
    paths = dataset.path.unique()
    train_paths, val_paths = train_test_split(paths, test_size=0.15)

    return (dataset[dataset['path'].isin(train_paths)],
           dataset[dataset['path'].isin(val_paths)])


def train(model, dataset : DataFrame, config : dict, neptune_project : str):
    seed_everything(base.SEED)

    epochs = config['epochs']
    batch_size = config['bs']
    output_dir = config['checkpoint_dir']

    train_dataset, val_dataset = split_dataset(dataset=dataset)
    train_dataloader = waste_detection_dataset.get_dataloader(
        data=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = waste_detection_dataset.get_dataloader(
        data=val_dataset, batch_size=1, shuffle=False
    )

    neptune_logger = NeptuneLogger(
        api_key=base.NEPTUNE_API_KEY,
        project=neptune_project
    )

    task = ModelModule(model=model, config=config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='Validation_mAP', 
        mode='max',
        dirpath=output_dir
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
            num_sanity_val_steps=0,
            max_epochs=epochs,
            logger=neptune_logger,
            accelerator='gpu', devices=1
        )
    else:
        trainer = Trainer(
            gpus=base.GPU,
            callbacks=[checkpoint_callback, learningrate_callback,
                earlystopping_callback],
            default_root_dir=Path(output_dir),
            num_sanity_val_steps=0,
            max_epochs=epochs,
            logger=neptune_logger
        )

    trainer.fit(model=task, train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader)
    
    log_packages_neptune(neptune_logger=neptune_logger)
    
    best_model_path = checkpoint_callback.best_model_path
    log_model_neptune(
        checkpoint_path=Path(best_model_path), 
        neptune_logger=neptune_logger,
        save_directory=Path(output_dir), 
        name='best_model.pt'
    )
    
    neptune_logger.experiment.stop()
    
    return best_model_path, trainer


def test(trainer : Trainer, dataset : DataFrame):
    dataloader = waste_detection_dataset.get_dataloader(
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
    print(checkpoint_path)
    print(save_directory)
    print(name)
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["hyper_parameters"]["model"]
    torch.save(model.state_dict(), save_directory / name)

    neptune_logger['model'].upload(File.as_pickle(model))