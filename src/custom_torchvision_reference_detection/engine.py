import math
import sys
import time, datetime

import torch
from ..custom_torchvision.models.detection import MaskRCNN, KeypointRCNN
from ..custom_torchvision_reference_detection import utils, coco_eval, coco_utils
from ..custom_ultralytics_yolov5.utils import loss
from ..custom_ultralytics_yolov5.models import yolo


def simple_train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, verbose=True):
    start_time = time.time()
    model = model.to(device)
    model.train()

    if model is yolo.Model:
        compute_loss = loss.ComputeLoss(model)  # init loss class

    accumulated_loss = 0

    segments = 25
    abs_completed = 0
    completed_at = len(data_loader)
    if verbose:
        print(f'[Epoch {epoch:3}] {"█"*int(abs_completed*segments/completed_at)}{"░"*(segments-int(abs_completed*segments/completed_at))} ⏳', end='\r')

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=False):
            if model is yolo.Model:
                preds = model(images)
                yolo_loss, _yolo_loss_items = compute_loss(preds, targets.to(device))  # loss scaled by batch_size
                loss_value = yolo_loss
                losses_reduced = yolo_loss
            else:
                loss_dict, _dets = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()
        accumulated_loss += loss_value

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        abs_completed += 1
        if verbose:
            print(f'[Epoch {epoch:3}] {"█"*int(abs_completed*segments/completed_at)}{"░"*(segments-int(abs_completed*segments/completed_at))} ⏳', end='\r')

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    lr_scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if verbose:
        print(f'[Epoch {epoch:3}] {"█"*int(abs_completed*segments/completed_at)}{"░"*(segments-int(abs_completed*segments/completed_at))} ✅ [{total_time_str}]', end='')

    accumulated_loss = accumulated_loss / len(data_loader)

    return model, optimizer, lr_scheduler, accumulated_loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model = model.to(device)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = coco_utils.get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = coco_eval.CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
