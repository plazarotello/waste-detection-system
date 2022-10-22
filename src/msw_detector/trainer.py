# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from typing import List, Dict, Union
import os
import re
import time
import datetime
from pathlib import Path

from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import custom_torchvision
from custom_torchvision import transforms

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from custom_torchvision_reference_detection import utils, engine

from msw_detector import shared_data as base


def custom_collate(data):
    images = []
    targets = []
    for image, target in data:
        images.append(image)
        targets.append(target)
    return images, targets


class RecyclingDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, data_augmentation : bool = False):
        'Initialization'
        self.images = images
        self.targets = targets
        self.data_augment = data_augmentation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        return (path_to_image(self.images[index], self.data_augment), self.targets[index])


# -----------------------------------------------------------------------------

def path_to_image(path, augment : bool = False):
    t = transforms.Compose([transforms.ToTensor()])
    if augment:
        t = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.25), transforms.RandomRotation((-15, 15))])
    img = Image.open(path)
    img = img.convert('RGB')
    return t(img)


def paths_to_images(paths: List):
    return [path_to_image(_path) for _path in paths]


# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return transforms.ToPILImage()(img).convert('RGB')

# -----------------------------------------------------------------------------

def split_dataset(dataset: DataFrame, binary_classification : bool):
    grouped_df = dataset.groupby(['path'])
    train_paths, val_paths = train_test_split(
        dataset['path'].unique(), test_size=0.1)

    train_targets = []
    val_targets = []

    for tpath in train_paths:
        train_bbs = []
        train_labels = []
        for _, row in grouped_df.get_group(tpath).iterrows():
            width = float(row['width'])
            height = float(row['height'])
            label = row['label']
            train_bbs.append([row['bbox-x']/width, row['bbox-y']/height,
                              (row['bbox-x']+row['bbox-w'])/width,
                              (row['bbox-y']+row['bbox-h'])/height])
            if binary_classification:
                train_labels.append(1)
            else:
                train_labels.append(base.IDX_CATS[label])
        train_targets.append({'boxes': torch.as_tensor(train_bbs, dtype=torch.float32),
                              'labels': torch.as_tensor(train_labels, dtype=torch.int64)})

    for vpath in val_paths:
        val_bbs = []
        val_labels = []
        for _, row in grouped_df.get_group(vpath).iterrows():
            width = float(row['width'])
            height = float(row['height'])
            val_bbs.append([row['bbox-x']/width, row['bbox-y']/height,
                            (row['bbox-x']+row['bbox-w'])/width,
                            (row['bbox-y']+row['bbox-h'])/height])
            if binary_classification:
                val_labels.append(1)
            else:
                val_labels.append(base.IDX_CATS[row['label']])
        val_targets.append({'boxes': torch.as_tensor(val_bbs, dtype=torch.float32),
                            'labels': torch.as_tensor(val_labels, dtype=torch.int64)})

    return train_paths, train_targets, val_paths, val_targets


def evaluate(model, data_loader, device):
    model.to(device)
    accumulated_loss = 0
    metric = MeanAveragePrecision(box_format='xywh')

    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            image = torch.squeeze(image).to(device)
            target['boxes'] = torch.squeeze(target['boxes'], 0).to(device)
            target['labels'] = torch.squeeze(target['labels'], 0).to(device)

            loss_dict, detections = model.forward([image], [target])
            losses = sum(loss for loss in loss_dict.values())

            accumulated_loss += (losses.item() if isinstance(losses, torch.Tensor) else losses) 
            metric.update([apply_nms(detections[0])], [target])

    accumulated_loss = accumulated_loss / len(data_loader)

    metrics = metric.compute()
    mean_average_precision = metrics['map']
    # average recall with 10 detections per image
    mean_average_recall = metrics['mar_10']
    model.train()

    return (accumulated_loss, mean_average_precision.item(), mean_average_recall.item(),
            len(data_loader))

# -----------------------------------------------------------------------------

# https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch
# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = custom_torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction



# -------------------------------------------------------------------------------

def test(model, images: list, device) -> list[tuple[Path, list[Dict]]]:
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for image in images:
            img = torch.unsqueeze(path_to_image(image), 0).to(device)
            detections = apply_nms(model(img)[0])
            results.append((image, detections))
    return results

# -----------------------------------------------------------------------------


def plot_training_results(model_folder: Union[Path, str], figsize: tuple = (15, 5)):
    regex = re.compile('model_.+.pth')
    available_checkpoints = [os.path.join(
        model_folder, f) for f in os.listdir(model_folder) if regex.match(f)]

    acc_loss = {'train': [], 'val': []}
    mAP = []
    mAR = []

    epochs = range(1, len(available_checkpoints)+1)

    checkpoint = torch.load(os.path.join(
        model_folder, f'model_{len(available_checkpoints)-1}.pth'))

    # only store accumulated loss, mAP, mAR
    acc_loss['train'] = [item[0].item()
                         for item in checkpoint['train_results']]
    
    acc_loss['val'] = [item[0].item() for item in checkpoint['val_results']]
    mAP = [item[1].item() for item in checkpoint['val_results']]
    mAR = [item[2].item() for item in checkpoint['val_results']]

    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['figure.titlesize'] = 16
    sns.set_theme()

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize, dpi=300)
    fig.suptitle('Gráficos de diagnóstico')

    metrics_ax = (axes.flat)[0]
    metrics_ax.plot(epochs, mAP, label=f'mAP (validation)')
    metrics_ax.plot(epochs, mAR, label=f'mAR (validation)')
    metrics_ax.set(title='Métricas', xlabel='epoch',
                   ylabel='precision/recall', xticks=epochs)
    metrics_ax.set_ylim([0.0, 1.0])
    metrics_ax.legend(loc='upper right')

    loss_ax = (axes.flat)[1]
    loss_ax.plot(epochs, acc_loss['train'], label=f'loss (train)')
    loss_ax.plot(epochs, acc_loss['val'], label=f'loss (validation)')
    loss_ax.set(title='Loss plot', xlabel='epoch',
                ylabel='loss', xticks=epochs)
    loss_ax.legend(loc='upper right')

    if len(epochs) > 50:
        epoch_gap = 10
    else:
        epoch_gap = 1
    metrics_ax.set_xticks(metrics_ax.get_xticks()[::epoch_gap])
    loss_ax.set_xticks(loss_ax.get_xticks()[::epoch_gap])

    fig.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------

def choose_best_model(models_dir : str, base_model):
    regex = re.compile('model_.+.pth')
    available_checkpoints = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if regex.match(f)]
    last_checkpoint = torch.load(os.path.join(models_dir, f'model_{len(available_checkpoints)-1}.pth'))
    mAP = [item[1].item() for item in last_checkpoint['val_results']]
    index_max_map = max(range(len(mAP)), key=mAP.__getitem__)
    best_checkpoint = torch.load(os.path.join(models_dir, f'model_{index_max_map}.pth'))
    base_model.load_state_dict(best_checkpoint['model'], strict=False)
    print(f'Best model: model{index_max_map}.pth')
    return base_model
# -----------------------------------------------------------------------------

def train(model, dataset: DataFrame, bs: int, optimizer: optim.Optimizer,
          lr_scheduler: lr_scheduler, epochs: int, output_dir: str,
          device, augment : bool = False, resume: bool = True, 
          binary_classification : bool = False, save : bool = True):
    torch.cuda.empty_cache()

    model_without_ddp = model

    train_imgs, train_targets, val_imgs, val_targets = \
        split_dataset(dataset, binary_classification)
    train_dataset = RecyclingDataset(train_imgs, train_targets, 
        data_augmentation=augment)
    val_dataset = RecyclingDataset(val_imgs, val_targets, data_augmentation=False)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, \
        shuffle=True, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    if resume:
        regex = re.compile('model_.+.pth')
        available_checkpoints = [os.path.join(
            output_dir, f) for f in os.listdir(output_dir) if regex.match(f)]
        latest_checkpoint = max(available_checkpoints, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        loss_train = checkpoint['train_results']
        acc_val = checkpoint['val_results']
    else:
        start_epoch = 0
        loss_train = []
        acc_val = []

    if start_epoch >= epochs:
        return model


    start_time = time.time()
    # train for n epochs
    for epoch in range(start_epoch, epochs):
        model, optimizer, lr_scheduler, loss = engine.simple_train_one_epoch(
            model, optimizer, lr_scheduler, train_dataloader, device, epoch)

        loss_train.append(loss)
        val_results = evaluate(model, val_dataloader, device)
        acc_val.append(val_results)

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_results': loss_train,
            'val_results': acc_val,
            'epoch': epoch,
        }
        if save:
            utils.save_on_master(checkpoint, os.path.join(
                output_dir, f'model_{epoch}.pth'))

        print(f' LR:{lr_scheduler.get_last_lr()[0]:1.6} | Loss:{round(loss, 2):2.2}-{round(val_results[0], 2):2.2} | '
            f'mAP:{round(val_results[1], 2):1.2} | mAR:{round(val_results[2], 2):1.2}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')
    return model, loss_train, acc_val
# -----------------------------------------------------------------------------