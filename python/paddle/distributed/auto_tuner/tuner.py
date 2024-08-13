# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import csv
import os

from .utils import default_candidates, gbs_default_candidates


class AutoTuner:
    """
    The AutoTuner can automatically provide running task based on user-defined settings
    and the task will be launched for execution.

    Args:
        tuner_cfg (dict): The configuration of auto tuner user defined.
    """

    def __init__(self, tuner_cfg):
        self.cur_task_id = 1
        self.task_limit = tuner_cfg.get("task_limit", 100)

        search_algo = tuner_cfg.get("search_algo", {"name": "grid"})["name"]

        if search_algo == "grid":
            from .search import GridSearch

            tuner_cfg["candidates"] = default_candidates(tuner_cfg)
            self.algo = GridSearch(tuner_cfg)
        elif search_algo == "dp_estimation":
            from .search import DpEstimationSearch

            tuner_cfg["candidates"] = default_candidates(tuner_cfg)
            self.algo = DpEstimationSearch(tuner_cfg)
        elif search_algo == "gbs":
            from .search import GBSSearch

            tuner_cfg["candidates"] = gbs_default_candidates(tuner_cfg)
            self.algo = GBSSearch(tuner_cfg)
        elif search_algo == "customize":
            from .search import CustomizeSearch

            self.algo = CustomizeSearch(tuner_cfg)
        else:
            raise NotImplementedError

        self.history_cfgs = []
        self.resume_cfgs = []
        self.tuner_cfg = tuner_cfg

    def search_once(self):
        """Return a new task config."""
        if self.cur_task_id > self.task_limit:
            return None
        new_cfg = self.algo.search_once(self.history_cfgs)
        self.cur_task_id += 1

        return new_cfg

    def add_cfg(self, cfg):
        """Add cfg into history cfgs"""
        self.history_cfgs.append(cfg)

    def resume_form_history(self, history_csv_path="./history.csv"):
        """Resume form history csv file"""
        # The breakpoint resume function does not start when the resume csv file does not exist.
        if not os.path.exists(history_csv_path):
            return
        resume_csv_path = os.path.join(
            os.path.dirname(history_csv_path),
            f'{os.path.basename(history_csv_path).split(".")[0]}_copy.csv',
        )
        with open(history_csv_path, "r") as fread:
            reader = csv.reader(fread)
            data_list = list(reader)
            with open(resume_csv_path, "w") as fwrite:
                writer = csv.writer(fwrite)
                for row in data_list:
                    writer.writerow(row)
        # chang str type to real type
        for row in data_list:
            for i, value in enumerate(row):
                try:
                    row[i] = int(value)
                except ValueError:
                    try:
                        row[i] = float(value)
                    except ValueError:
                        pass

        data_dict = []
        keys = data_list[0]
        values = data_list[1:]
        for val in values:
            val = [x if x != '' else None for x in val]
            val = [True if x == 'True' else x for x in val]
            val = [False if x == 'False' else x for x in val]
            dictionary = dict(zip(keys, val))
            time_val = -1
            target_key = self.tuner_cfg["metric_cfg"]["name"]
            if dictionary[target_key]:
                time_val = dictionary[target_key]
            dictionary["time"] = time_val
            data_dict.append(dictionary)
        self.resume_cfgs = data_dict

    def get_cfg_from_resume(self, cur_cfg):
        """Get cfg from resume cfgs"""
        keys_to_compare = [
            'mp_degree',
            'sharding_degree',
            'pp_degree',
            'dp_degree',
            'sharding_stage',
            'micro_batch_size',
            'vpp_degree',
            'use_recompute',
            'recompute_granularity',
            'num_gpus',
            'nodes',
            'global_batch_size',
            'sharding_overlap',
            'acc_steps',
        ]

        if self.tuner_cfg.get("refined_recompute", None):
            for rr in self.tuner_cfg["refined_recompute"]:
                keys_to_compare.append(rr)

        if self.tuner_cfg.get("custom_search_dim", None):
            for key in self.tuner_cfg["custom_search_dim"]:
                keys_to_compare.append(key)

        for cfg in self.resume_cfgs:
            ret_is_same = True
            for key in keys_to_compare:
                if not cfg.get(key) and not cur_cfg.get(key):
                    continue
                else:
                    is_same = str(cfg.get(key)) == str(cur_cfg.get(key))
                ret_is_same = ret_is_same and is_same
            if ret_is_same:
                return cfg
        return None
