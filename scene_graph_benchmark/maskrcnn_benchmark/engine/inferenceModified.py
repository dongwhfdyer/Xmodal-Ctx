# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import imp
import logging
import time
import os
import json
import base64

import h5py
import numpy
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


class Features2HDF5:
    def __init__(self, objPath):
        if os.path.isfile(objPath):
            os.remove(objPath)
        self.objPath = objPath

        # print abs path
        print("obj file created in: ", os.path.abspath(objPath))

    def saveOneInstance(self, img_id, numBoxes, objFeatures):
        obj = h5py.File(self.objPath, 'a')
        obj.create_group(img_id)
        obj[img_id]['num_boxes'] = numBoxes
        obj[img_id]['obj_features'] = objFeatures
        obj.close()

    def saveByBatch(self, featureDict):
        self.obj = h5py.File(self.objPath, 'a')
        for img_id, feature in featureDict.items():
            img_id = str(img_id)
            self.obj.create_group(img_id)
            self.obj[img_id]['num_boxes'] = len(feature.get_field('box_features'))
            self.obj[img_id]['obj_features'] = feature.get_field('box_features').numpy()
        self.obj.close()


class Features2HDF5_v2(Features2HDF5):
    def __init__(self, objPath):
        super().__init__(objPath)
        self.obj = h5py.File(self.objPath, 'w')

    def saveByBatch(self, featureDict):
        for img_id, feature in featureDict.items():
            img_id = str(img_id)
            self.obj.create_group(img_id)
            self.obj[img_id]['num_boxes'] = len(feature.get_field('box_features'))
            self.obj[img_id]['obj_features'] = feature.get_field('box_features').numpy()

    def close(self):
        self.obj.close()


def featureExtractor(model, data_loader, device, bbox_aug):
    model.eval()
    results_dict = {}
    fh = Features2HDF5(r"featureOutputs/puzzleCOCOFeature.hdf5")
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

        if batchInd % 1000 == 0:  # todo: first epoch should not be saved.
            fh.saveByBatch(results_dict)
            # #---------kkuhn-block------------------------------ only for testing
            # obj = h5py.File(r"temp/rubb.hdf5", 'r')
            # print(obj.keys())
            # import numpy
            # print(numpy.array(obj['133']['num_boxes']))
            # print(numpy.array(obj['133']['obj_features']))
            # obj.close()
            # #---------kkuhn-block------------------------------
            results_dict = {}
        # if batchInd == 6:
        #     #---------kkuhn-block------------------------------ only for testing
        #     obj = h5py.File(r"temp/rubb.hdf5", 'r')
        #     print(obj.keys())
        #     # import numpy
        #     # print(numpy.array(obj['133']['num_boxes']))
        #     # print(numpy.array(obj['133']['obj_features']))
        #     obj.close()
        #     #---------kkuhn-block------------------------------
        #     pass

        # #---------kkuhn-block------------------------------ early stop
        # if batchInd == 100:
        #     break
        # #---------kkuhn-block------------------------------
    if len(results_dict) > 0:
        fh.saveByBatch(results_dict)
        results_dict = {}
    fh.close()

    return results_dict


def save2hdf5(objects):
    import h5py
    obj_file = r"/home/pcl/kuhn/Xmodal-Ctx/m2/datasets/vinvl.hdf5"
    # obj_file = r"~/kuhn/Xmodal-Ctx/m2/datasets/vinvl.hdf5"
    obj = h5py.File(obj_file, "r")
    print("--------------------------------------------------")


def inference(
        model,
        cfg,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="0",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        eval_attributes=False,
        save_predictions=False,
        skip_performance_eval=False,
        labelmap_file='',
):
    device = torch.device("cuda")

    # ---------kkuhn-block------------------------------ inference
    featureExtractor(model, data_loader, device, bbox_aug)
    # ---------kkuhn-block------------------------------
