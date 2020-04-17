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

import numpy as np
import pandas as pd
import os
import sys
import json

sys.path.append('../')

from hapi.metrics import Metric
from bmn_utils import boundary_choose, bmn_post_processing


class BmnMetric(Metric):
    """
    only support update with batch_size=1
    """

    def __init__(self, cfg, mode):
        super(BmnMetric, self).__init__()
        self.cfg = cfg
        self.mode = mode
        #get video_dict and video_list
        if self.mode == 'test':
            self.get_test_dataset_dict()
            if not os.path.isdir(self.cfg.TEST.output_path):
                os.makedirs(self.cfg.TEST.output_path)
            if not os.path.isdir(self.cfg.TEST.result_path):
                os.makedirs(self.cfg.TEST.result_path)
        elif self.mode == 'infer':
            self.get_infer_dataset_dict()
            if not os.path.isdir(self.cfg.INFER.output_path):
                os.makedirs(self.cfg.INFER.output_path)
            if not os.path.isdir(self.cfg.INFER.result_path):
                os.makedirs(self.cfg.INFER.result_path)

    def add_metric_op(self, preds, label):
        pred_bm, pred_start, pred_en = preds
        video_index = label[-1]
        return [pred_bm, pred_start, pred_en, video_index]  #return list

    def update(self, pred_bm, pred_start, pred_end, fid):
        # generate proposals
        pred_start = pred_start[0]
        pred_end = pred_end[0]
        fid = fid[0]

        if self.mode == 'infer':
            output_path = self.cfg.INFER.output_path
        else:
            output_path = self.cfg.TEST.output_path
        tscale = self.cfg.MODEL.tscale
        dscale = self.cfg.MODEL.dscale
        snippet_xmins = [1.0 / tscale * i for i in range(tscale)]
        snippet_xmaxs = [1.0 / tscale * i for i in range(1, tscale + 1)]
        cols = ["xmin", "xmax", "score"]

        video_name = self.video_list[fid]
        pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
        start_mask = boundary_choose(pred_start)
        start_mask[0] = 1.
        end_mask = boundary_choose(pred_end)
        end_mask[-1] = 1.
        score_vector_list = []
        for idx in range(dscale):
            for jdx in range(tscale):
                start_index = jdx
                end_index = start_index + idx
                if end_index < tscale and start_mask[
                        start_index] == 1 and end_mask[end_index] == 1:
                    xmin = snippet_xmins[start_index]
                    xmax = snippet_xmaxs[end_index]
                    xmin_score = pred_start[start_index]
                    xmax_score = pred_end[end_index]
                    bm_score = pred_bm[idx, jdx]
                    conf_score = xmin_score * xmax_score * bm_score
                    score_vector_list.append([xmin, xmax, conf_score])

        score_vector_list = np.stack(score_vector_list)
        video_df = pd.DataFrame(score_vector_list, columns=cols)
        video_df.to_csv(
            os.path.join(output_path, "%s.csv" % video_name), index=False)

        return 0  # result has saved in output path

    def accumulate(self):
        return 'post_processing is required...'  # required method

    def reset(self):
        print("Post_processing....This may take a while")
        if self.mode == 'test':
            bmn_post_processing(self.video_dict, self.cfg.TEST.subset,
                                self.cfg.TEST.output_path,
                                self.cfg.TEST.result_path)
        elif self.mode == 'infer':
            bmn_post_processing(self.video_dict, self.cfg.INFER.subset,
                                self.cfg.INFER.output_path,
                                self.cfg.INFER.result_path)

    def name(self):
        return 'bmn_metric'

    def get_test_dataset_dict(self):
        anno_file = self.cfg.MODEL.anno_file
        annos = json.load(open(anno_file))
        subset = self.cfg.TEST.subset
        self.video_dict = {}
        for video_name in annos.keys():
            video_subset = annos[video_name]["subset"]
            if subset in video_subset:
                self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()

    def get_infer_dataset_dict(self):
        file_list = self.cfg.INFER.filelist
        annos = json.load(open(file_list))
        self.video_dict = {}
        for video_name in annos.keys():
            self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()
