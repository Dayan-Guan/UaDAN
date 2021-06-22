# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import numpy as np

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    # logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    # logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    # start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    # synchronize()
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=total_time))
    # logger.info(
    #     "Total inference time: {} ({} s / img per device, on {} devices)".format(
    #         total_time_str, total_time * num_devices / len(dataset), num_devices
    #     )
    # )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    # all_boxes = [[] for _ in range(9)]
    all_boxes = [[[] for _ in range(len(predictions))] for _ in range(21)]

    for ind, pred in enumerate(predictions):
        img_info = dataset.get_img_info(ind)
        if len(pred) == 0:
            continue

        image_width = img_info["width"]
        image_height = img_info["height"]
        pred = pred.resize((image_width, image_height))

        labels = pred.get_field("labels").numpy()
        scores = pred.get_field("scores").numpy()
        bbox = pred.bbox.numpy()

        for cls in np.unique(labels):
            box = bbox[labels==cls,:]
            score = scores[labels==cls][:, np.newaxis]
            gt = np.append(box,score,axis=1)
            all_boxes[cls][ind] = gt

    import pickle
    with open(output_folder + "/all_boxes.pkl", 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
