#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import yaml
import paddle.fluid as fluid
import logging
from paddle.distributed.utils import get_logger

__all__ = []
logger = get_logger(logging.INFO, name="metrics")

# read metric config from yaml and init MetricMsg in fleet_wrapper
def init_metric(metric_ptr,
                metric_yaml_path,
                cmatch_rank_var="",
                mask_var="",
                phase=-1,
                cmatch_rank_group="",
                ignore_rank=False,
                bucket_size=1000000):
    yaml_fobj = open(metric_yaml_path)
    if sys.version.startswith('2.7.13'):
        content = yaml.load(yaml_fobj)
    else:
        content = yaml.load(yaml_fobj, Loader = yaml.FullLoader)
    
    print("yaml metric config: \n")
    print(content)

    metric_runner_list = content['monitors']
    if not metric_runner_list:
        metric_runner_list = []

    for metric_runner in metric_runner_list:
        is_join = metric_runner['phase'] == 'JOINING'
        phase=1 if is_join else 0

        metric_ptr.init_metric(metric_runner['method'], 
                              metric_runner['name'],
                              metric_runner['label'], 
                              metric_runner['target'],
                              cmatch_rank_var,
                              mask_var,
                              phase,
                              cmatch_rank_group,
                              ignore_rank,
                              bucket_size)

        # if auc_runner['method'] == 'CmatchRankMaskAucCalculator':
        #     util.init_metric(box, auc_runner['method'], auc_runner['name'],
        #             auc_runner['label'], target, mask_var=auc_runner['mask'], phase=1 if is_join else 0,
        #             cmatch_rank_var="cmatch_rank",
        #             cmatch_rank_group=auc_runner['cmatch_group'], ignore_rank=True,
        #             sample_scale_var=sample_scale_var)
        # elif 'cmatch_rank_group' in auc_runner:
        #     util.init_metric(box, auc_runner['method'], auc_runner['name'],
        #             auc_runner['label'], target, phase=1 if is_join else 0,
        #             cmatch_rank_var="cmatch_rank",
        #             cmatch_rank_group=auc_runner['cmatch_rank_group'],
        #             sample_scale_var=sample_scale_var)
        # elif 'cmatch_group' in auc_runner:
        #     util.init_metric(box, 'CmatchRankAucCalculator', auc_runner['name'],
        #             auc_runner['label'], target, phase=1 if is_join else 0,
        #             cmatch_rank_var="cmatch_rank",
        #             cmatch_rank_group=auc_runner['cmatch_group'], ignore_rank=True,
        #             sample_scale_var=sample_scale_var)
        # elif 'mask' in auc_runner:
        #     util.init_metric(box, auc_runner['method'], auc_runner['name'],
        #             auc_runner['label'], target, mask_var=auc_runner['mask'], phase=1 if is_join else 0,
        #             sample_scale_var=sample_scale_var)
        # else:
        #     util.init_metric(box, auc_runner['method'], auc_runner['name'],
        #             auc_runner['label'], target, phase=1 if is_join else 0,
        #             sample_scale_var=sample_scale_var)


def print_metric(metric_ptr, name):
    """
    print the metric value. Print directly in back-end
    """
    metric = metric_ptr.get_metric_msg(name)
    monitor_msg = "%s: AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f RMSE=%.6f "\
            "Actual CTR=%.6f Predicted CTR=%.6f COPC=%.6f INS Count=%.0f"\
            % (name, metric[0], metric[1], metric[2], metric[3], metric[4],
                    metric[5], metric[6], metric[7])
    # logger.info(monitor_msg)
    return monitor_msg


def print_auc(metric_ptr, is_day, phase="all"):
    """
    print metric according to stage and phase
    """
    if is_day is True:
        stage = "day"
        stage_num = -1
    else:
        stage = "pass"
        stage_num = 1 if phase == "join" else 0
    metric_results = []

    name_list = metric_ptr.get_metric_name_list(stage_num)
    if phase == "all":
        for name in name_list:
            if name.find(stage) != -1:
                metric_results.append(print_metric(metric_ptr, name=name))
    else:
        for name in name_list:
            if name.find(stage) != -1 and name.find(phase) != -1:
                metric_results.append(print_metric(metric_ptr, name=name))

    return metric_results
