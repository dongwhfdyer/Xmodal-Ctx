# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import imp
import logging
import time
import os
import json
import base64

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.structures.tsv_file_ops import tsv_writer
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from scene_graph_benchmark.scene_parser import SceneParserOutputs

from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather, gather_on_master
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


def saveFeatures2hdf5(results_dict):
    pass


def compute_on_dataset(model, data_loader, device, bbox_aug):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for batchInd, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, scales = batch[0], batch[1], batch[2], batch[3:]
        with torch.no_grad():
            try:
                output = model(images.to(device), targets)
            except RuntimeError as e:
                image_ids_str = [str(img_id) for img_id in image_ids]
                print("Runtime error occurred in Image Ids: {}"
                      .format(','.join(image_ids_str)))
                print(e)
                continue
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )

        if batchInd == 10:
            saveFeatures2hdf5(results_dict)
            results_dict = {}
    if len(results_dict) > 0:
        saveFeatures2hdf5(results_dict)
        results_dict = {}

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, gather_on_cpu=False):
    if gather_on_cpu:  # True
        all_predictions = gather_on_master(predictions_per_gpu)
    else:
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

    return predictions


def save2hdf5(objects):
    import h5py
    obj_file = r"/home/pcl/kuhn/Xmodal-Ctx/m2/datasets/vinvl.hdf5"
    # obj_file = r"~/kuhn/Xmodal-Ctx/m2/datasets/vinvl.hdf5"
    obj = h5py.File(obj_file, "r")
    print("--------------------------------------------------")


def convert_predictions_to_tsv(predictions, dataset, output_folder,
                               data_subset, labelmap_file=None,  # data_subset = ['rect', class, 'conf', 'feature']
                               relation_on=False,
                               output_tsv_name='predictions.tsv'):
    # convert the prediction results to tsv format and save
    # for easier visualization and post-processing.

    # # ---------kkuhn-block------------------------------# Kuhn commented
    # if 'class' in data_subset:
    #     if os.path.isfile(labelmap_file):
    #         labelmap = load_labelmap_file(labelmap_file)
    #         labelmap = {labelmap[key] + 1: key for key in labelmap}
    #     elif hasattr(dataset, 'ind_to_class'):
    #         labelmap = dataset.ind_to_class
    #     else:
    #         raise ValueError("object labelmap is required, but was not provided")
    # # ---------kkuhn-block------------------------------

    if 'attr_labels' in data_subset:
        if os.path.isfile(labelmap_file):
            attr_labelmap = json.load(open(labelmap_file, 'r'))['attribute_to_idx']
            attr_labelmap['__no_attribute__'] = 0
            attr_labelmap = {v: k for k, v in attr_labelmap.items()}
        elif hasattr(dataset, 'ind_to_attribute'):
            attr_labelmap = dataset.ind_to_attribute
        else:
            raise ValueError("attribute labelmap is required, but was not provided")
    if 'relations' in data_subset:
        if os.path.isfile(labelmap_file):
            relation_labelmap = json.load(open(labelmap_file, 'r'))['predicate_to_idx']
            relation_labelmap['__no_relation__'] = 0
            relation_labelmap = {relation_labelmap[key]: key for key in relation_labelmap}
        elif hasattr(dataset, 'ind_to_relation'):
            relation_labelmap = dataset.ind_to_relation
        else:
            raise ValueError("relation labelmap is required, but was not provided")

    def gen_rows():
        for idx, prediction in sorted(predictions.items()):
            # image_key = dataset.get_img_key(idx) # kuhn edited. This statement will raise an error.
            image_width = dataset.get_img_info(idx)['width']
            image_height = dataset.get_img_info(idx)['height']

            if isinstance(prediction, SceneParserOutputs):
                prediction_pred = prediction.prediction_pairs
                prediction = prediction.predictions

                relations = prediction_pred.get_field("idx_pairs").numpy()
                relation_scores = prediction_pred.get_field("scores").numpy()
                predicates = prediction_pred.get_field("labels").numpy()
                if 'relation_scores_all' in data_subset:
                    relation_scores_all = prediction_pred.get_field("scores_all").numpy()
                if 'relation_feature' in data_subset:
                    relation_features = prediction_pred.get_field("pred_features").numpy()

            prediction = prediction.resize((image_width, image_height))
            boxes = prediction.bbox.tolist()

            if 'conf' in data_subset:
                scores = prediction.get_field('scores').tolist()
            if 'class' in data_subset:
                labels = prediction.get_field('labels').tolist()
            if 'feature' in data_subset:
                features = prediction.get_field('box_features').numpy()
            if 'scores_all' in data_subset:
                scores_all = prediction.get_field('scores_all').numpy()
            if 'boxes_all' in data_subset:
                boxes_all = prediction.get_field('boxes_all').numpy()
            if "attr_labels" in data_subset:
                attr_labels = prediction.get_field("attr_labels").tolist()
            if "attr_scores" in data_subset:
                attr_scores = prediction.get_field("attr_scores").tolist()
            if "attr_scores_all" in data_subset:
                attr_scores_all = prediction.get_field("attr_scores_all").numpy()
            if 'relations' in data_subset:
                relations = relations.tolist()
                predicates = [relation_labelmap[rel + 1] for rel in predicates.tolist()]
            if 'relation_scores' in data_subset:
                relation_scores = relation_scores.tolist()
            if 'relation_scores_all' in data_subset:
                relation_scores_all = [base64.b64encode(relation_scores_all[i]).decode('utf-8') for i in range(len(relations))]

            objects = []
            for i in range(len(boxes)):
                cur_d = {}
                for name in data_subset:
                    if name == 'rect':
                        cur_d['rect'] = boxes[i]
                        cur_d['bbox_id'] = i
                    if name == 'class':
                        cur_d['class'] = labelmap[labels[i]]
                    if name == 'conf':
                        cur_d['conf'] = scores[i]
                    if name == 'feature':
                        # cur_d['feature'] = base64.b64encode(features[i]).decode('utf-8') # kuhn edited
                        cur_d['feature'] = features[i]
                    if name == 'scores_all':
                        cur_d['scores_all'] = base64.b64encode(scores_all[i]) \
                            .decode('utf-8')
                    if name == 'boxes_all':
                        cur_d['boxes_all'] = base64.b64encode(boxes_all[i]) \
                            .decode('utf-8')
                    if name == 'attr_labels':
                        cur_d['attributes'] = []
                        for attr in attr_labels[i]:
                            cur_d['attributes'].append(attr_labelmap[attr])
                    if name == 'attr_scores':
                        cur_d['attr_scores'] = []
                        for attr_score in attr_scores[i]:
                            cur_d['attr_scores'].append(attr_score)
                    if name == 'attr_scores_all':
                        cur_d['attr_scores_all'] = base64.b64encode(attr_scores_all[i]) \
                            .decode('utf-8')
                objects.append(cur_d)

            triplets = None
            if relation_on:  # False
                triplets = []
                for i in range(len(relations)):
                    cur_d = {}
                    for name in data_subset:
                        if name == 'relations':
                            cur_d['subj_id'] = relations[i][0]
                            cur_d['obj_id'] = relations[i][1]
                            cur_d['class'] = predicates[i]
                        if name == 'relation_scores':
                            cur_d['conf'] = relation_scores[i]
                        if name == 'relation_scores_all':
                            cur_d['scores_all'] = relation_scores_all[i]
                        if name == 'relation_feature':
                            cur_d['relation_feature'] = base64.b64encode(relation_features[i]).decode('utf-8')
                    triplets.append(cur_d)

            # yield json.dumps({'objects': objects, 'relations': triplets}) # kuhn commented
            save2hdf5(objects)
            # yield image_key, json.dumps({'objects': objects, 'relations': triplets}) # kuhn edited

    gen_rows()

    tsv_writer(gen_rows(), os.path.join(output_folder, output_tsv_name))  # kuhn commented


def inference(
        model,
        cfg,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        eval_attributes=False,
        save_predictions=False,
        skip_performance_eval=False,
        labelmap_file='',
):
    device = torch.device(device)
    dataset = data_loader.dataset

    # ---------kkuhn-block------------------------------ inference
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug )
    # ---------kkuhn-block------------------------------
