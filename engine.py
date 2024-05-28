# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import cv2
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, AP_path):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    sys.stdout = open(AP_path, 'w')
    coco_evaluator.summarize()
    sys.stdout.close()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

################## Inference functions ##################

def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().strip().split('\n')
    return {i: label for i, label in enumerate(labels)}

@torch.no_grad()
def preprocess(frame, device):
    # Example preprocessing (adjust as needed)
    # Convert frame to tensor, handle normalization, resizing, etc.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame).float() / 255.0  # Normalize to [0, 1]
    frame = frame.permute(2, 0, 1).unsqueeze(0).to(device)  # CHW format and add batch dimension
    return frame

@torch.no_grad()
def perform_inference(frame, model, input_tensor, postprocessors, device):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Dummy targets for processing (if necessary)
    targets = [{'orig_size': torch.tensor([frame.shape[0], frame.shape[1]]), 
                'size': torch.tensor([input_tensor.shape[2], input_tensor.shape[3]])}]
    # Process outputs
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
    results = postprocessors['bbox'](outputs, orig_target_sizes)
    return results

def draw_detections(frame, results, classes_path):
    LABELS = load_labels(classes_path)
    # Example of results handling -- adjust based on your actual output structure
    for result in results:
        boxes = result['boxes']  # Assuming the result dict contains 'boxes'
        scores = result['scores']  # Assuming there are confidence scores
        labels = result['labels']  # Assuming there are labels

        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5:  # Threshold can be adjusted
                label_name = LABELS.get(label.item(), "Unknown")  # Get the label name from the dictionary
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'{label_name}: {score:.2f}', (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame
    
@torch.no_grad()
def image_inference(frame, model, postprocessors, device, classes_path):
    input_tensor = preprocess(frame, device)
    results = perform_inference(frame, model, input_tensor, postprocessors, device)
    frame = draw_detections(frame, results, classes_path)
    return frame

@torch.no_grad()
def video_inference(input_video_path, model, postprocessors, device, output_video_path, classes_path):
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust codec as per your requirement
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    output_img_dir = os.path.join(os.path.dirname(output_video_path), os.path.basename(output_video_path).replace('.mp4', '') + '_images')
    os.makedirs(output_img_dir, exist_ok=True)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = image_inference(frame, model, postprocessors, device, classes_path)
        out.write(frame)
        
        if count % 100 == 0:
            cv2.imwrite(os.path.join(output_img_dir, f'{count}.png'), frame)
        count += 1
        
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
