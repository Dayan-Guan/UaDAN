import logging

from .voc_eval import do_voc_evaluation


def voc_evaluation(dataset, predictions, output_folder, box_only, **_):
    return do_voc_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
    )
