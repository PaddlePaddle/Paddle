#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import numpy as np
import json
import logging
import os
import sys

sys.path.append('../')

from distributed import DistributedBatchSampler
from paddle.fluid.io import Dataset, DataLoader

logger = logging.getLogger(__name__)

from config_utils import *
from bmn_utils import iou_with_anchors, ioa_with_anchors

DATATYPE = "float32"


class BmnDataset(Dataset):
    def __init__(self, cfg, mode):
        self.mode = mode
        self.tscale = cfg.MODEL.tscale  # 100
        self.dscale = cfg.MODEL.dscale  # 100
        self.anno_file = cfg.MODEL.anno_file
        self.feat_path = cfg.MODEL.feat_path
        self.file_list = cfg.INFER.filelist
        self.subset = cfg[mode.upper()]['subset']
        self.tgap = 1. / self.tscale

        self.get_dataset_dict()
        self.get_match_map()

    def __getitem__(self, index):
        video_name = self.video_list[index]
        video_idx = self.video_list.index(video_name)
        video_feat = self.load_file(video_name)
        if self.mode == 'infer':
            return video_feat, video_idx
        else:
            gt_iou_map, gt_start, gt_end = self.get_video_label(video_name)
            if self.mode == 'train' or self.mode == 'valid':
                return video_feat, gt_iou_map, gt_start, gt_end
            elif self.mode == 'test':
                return video_feat, gt_iou_map, gt_start, gt_end, video_idx

    def __len__(self):
        return len(self.video_list)

    def get_dataset_dict(self):
        assert (
            os.path.exists(self.feat_path)), "Input feature path not exists"
        assert (os.listdir(self.feat_path)), "No feature file  in feature path"
        self.video_dict = {}
        if self.mode == "infer":
            annos = json.load(open(self.file_list))
            for video_name in annos.keys():
                self.video_dict[video_name] = annos[video_name]
        else:
            annos = json.load(open(self.anno_file))
            for video_name in annos.keys():
                video_subset = annos[video_name]["subset"]
                if self.subset in video_subset:
                    self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()
        print("%s subset video numbers: %d" %
              (self.subset, len(self.video_list)))
        video_name_set = set(
            [video_name + '.npy' for video_name in self.video_list])
        assert (video_name_set.intersection(set(os.listdir(self.feat_path))) ==
                video_name_set), "Input feature not exists in feature path"

    def get_match_map(self):
        match_map = []
        for idx in range(self.tscale):
            tmp_match_window = []
            xmin = self.tgap * idx
            for jdx in range(1, self.tscale + 1):
                xmax = xmin + self.tgap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)
        match_map = np.transpose(match_map, [1, 0, 2])
        match_map = np.reshape(match_map, [-1, 2])
        self.match_map = match_map
        self.anchor_xmin = [self.tgap * i for i in range(self.tscale)]
        self.anchor_xmax = [self.tgap * i for i in range(1, self.tscale + 1)]

    def get_video_label(self, video_name):
        video_info = self.video_dict[video_name]
        video_second = video_info['duration_second']
        video_labels = video_info['annotations']

        gt_bbox = []
        gt_iou_map = []
        for gt in video_labels:
            tmp_start = max(min(1, gt["segment"][0] / video_second), 0)
            tmp_end = max(min(1, gt["segment"][1] / video_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.dscale, self.tscale])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)

        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_len_small = 3 * self.tgap
        gt_start_bboxs = np.stack(
            (gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack(
            (gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        match_score_start = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_start.append(
                np.max(
                    ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[
                        jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_end.append(
                np.max(
                    ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[
                        jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))

        gt_start = np.array(match_score_start)
        gt_end = np.array(match_score_end)
        return gt_iou_map.astype(DATATYPE), gt_start.astype(
            DATATYPE), gt_end.astype(DATATYPE)

    def load_file(self, video_name):
        file_name = video_name + ".npy"
        file_path = os.path.join(self.feat_path, file_name)
        video_feat = np.load(file_path)
        video_feat = video_feat.T
        video_feat = video_feat.astype("float32")
        return video_feat
