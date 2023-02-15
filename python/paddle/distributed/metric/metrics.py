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

import logging

import yaml

from paddle.distributed.utils.log_utils import get_logger

__all__ = []
logger = get_logger(logging.INFO, name="metrics")


# read metric config from yaml and init MetricMsg in fleet_wrapper
def init_metric(
    metric_ptr,
    metric_yaml_path,
    cmatch_rank_var="",
    mask_var="",
    uid_var="",
    phase=-1,
    cmatch_rank_group="",
    ignore_rank=False,
    bucket_size=1000000,
):
    yaml_fobj = open(metric_yaml_path)

    content = yaml.load(yaml_fobj, Loader=yaml.FullLoader)

    print("yaml metric config: \n")
    print(content)

    metric_runner_list = content['monitors']
    if not metric_runner_list:
        metric_runner_list = []

    for metric_runner in metric_runner_list:
        is_join = metric_runner['phase'] == 'JOINING'
        phase = 1 if is_join else 0

        if metric_runner['method'] == 'AucCalculator':
            metric_ptr.init_metric(
                metric_runner['method'],
                metric_runner['name'],
                metric_runner['label'],
                metric_runner['target'],
                cmatch_rank_var,
                mask_var,
                uid_var,
                phase,
                cmatch_rank_group,
                ignore_rank,
                bucket_size,
            )
        elif metric_runner['method'] == 'MultiTaskAucCalculator':
            metric_ptr.init_metric(
                metric_runner['method'],
                metric_runner['name'],
                metric_runner['label'],
                metric_runner['target'],
                metric_runner['cmatch_var'],
                mask_var,
                uid_var,
                phase,
                metric_runner['cmatch_group'],
                ignore_rank,
                bucket_size,
            )
        elif metric_runner['method'] == 'CmatchRankAucCalculator':
            metric_ptr.init_metric(
                metric_runner['method'],
                metric_runner['name'],
                metric_runner['label'],
                metric_runner['target'],
                metric_runner['cmatch_var'],
                mask_var,
                uid_var,
                phase,
                metric_runner['cmatch_group'],
                metric_runner['ignore_rank'],
                bucket_size,
            )
        elif metric_runner['method'] == 'MaskAucCalculator':
            metric_ptr.init_metric(
                metric_runner['method'],
                metric_runner['name'],
                metric_runner['label'],
                metric_runner['target'],
                cmatch_rank_var,
                metric_runner['mask'],
                uid_var,
                phase,
                cmatch_rank_group,
                ignore_rank,
                bucket_size,
            )
        elif metric_runner['method'] == 'CmatchRankMaskAucCalculator':
            metric_ptr.init_metric(
                metric_runner['method'],
                metric_runner['name'],
                metric_runner['label'],
                metric_runner['target'],
                metric_runner['cmatch_var'],
                metric_runner['mask'],
                uid_var,
                phase,
                metric_runner['cmatch_group'],
                metric_runner['ignore_rank'],
                bucket_size,
            )
        elif metric_runner['method'] == 'WuAucCalculator':
            metric_ptr.init_metric(
                metric_runner['method'],
                metric_runner['name'],
                metric_runner['label'],
                metric_runner['target'],
                cmatch_rank_var,
                mask_var,
                metric_runner['uid'],
                phase,
                cmatch_rank_group,
                ignore_rank,
                bucket_size,
            )
        else:
            metric_ptr.init_metric(
                metric_runner['method'],
                metric_runner['name'],
                metric_runner['label'],
                metric_runner['target'],
                cmatch_rank_var,
                mask_var,
                phase,
                cmatch_rank_group,
                ignore_rank,
                bucket_size,
            )


def print_metric(metric_ptr, name):
    """
    print the metric value. Print directly in back-end
    """
    if name.find("wuauc") != -1:
        metric = metric_ptr.get_wuauc_metric_msg(name)
        monitor_msg = (
            "%s: User Count=%.0f INS Count=%.0f UAUC=%.6f WUAUC=%.6f "
            % (name, metric[0], metric[1], metric[4], metric[5])
        )
    else:
        metric = metric_ptr.get_metric_msg(name)
        monitor_msg = (
            "%s: AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f RMSE=%.6f "
            "Actual CTR=%.6f Predicted CTR=%.6f COPC=%.6f INS Count=%.0f"
            % (
                name,
                metric[0],
                metric[1],
                metric[2],
                metric[3],
                metric[4],
                metric[5],
                metric[6],
                metric[7],
            )
        )
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
