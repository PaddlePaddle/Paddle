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

import copy
import csv
import os
from typing import Tuple

import pandas as pd


class HistoryRecorder:
    # NOTE increase extenable ablitity
    def __init__(self, tuner_cfg) -> None:
        self.tuner_cfg = tuner_cfg
        self.search_algo = self.tuner_cfg['search_algo']['name']
        self.history = []
        self.store_path = None
        self.additional_metric_key = None

    def add_cfg(self, **kwargs):
        cur_configs = {}
        for key, val in kwargs.items():
            cur_configs[key] = val
        self.history.append(cur_configs)

    def sort_metric(self, direction, metric_name) -> None:
        if direction == 'Maximize':
            self.history.sort(
                key=lambda x: x[metric_name]
                if x[metric_name] is not None
                else float('-inf'),
                reverse=True,
            )
        else:
            self.history.sort(
                key=lambda x: x[metric_name]
                if x[metric_name] is not None
                else float('inf'),
                reverse=False,
            )

    def get_best(
        self, metric, direction, buffer=None, max_mem_usage=None
    ) -> Tuple[dict, bool]:
        self.sort_metric(direction=direction, metric_name=metric)
        if len(self.history) == 0:
            return (None, True)

        best_cfg = self.history[0]
        if isinstance(best_cfg["max_mem_usage"], str) or best_cfg["time"] == -1:
            return (best_cfg, True)

        if buffer is not None:
            if buffer < 0:
                raise ValueError("The buffer should be not less than 0.")
            assert (
                max_mem_usage is not None
            ), "max_mem_usage cannot be None when buffer is greater than 0."
            if max_mem_usage <= 0:
                raise ValueError("max_mem_usage should be greater than 0.")

            for cfg in self.history:
                if (
                    not best_cfg["max_mem_usage"]
                    and cfg["max_mem_usage"]
                    and not isinstance(cfg["max_mem_usage"], str)
                    and cfg["time"] != -1
                ):
                    best_cfg = cfg
                    continue

                if (
                    not isinstance(cfg["max_mem_usage"], str)
                    and cfg["max_mem_usage"]
                    and cfg["max_mem_usage"] < best_cfg["max_mem_usage"]
                    and cfg["time"] != -1
                ):
                    best_cfg = cfg

                if (
                    not isinstance(cfg["max_mem_usage"], str)
                    and cfg["max_mem_usage"]
                    and cfg["max_mem_usage"] < max_mem_usage - buffer
                    and cfg["time"] != -1
                ):
                    break
            return (best_cfg, False)

        return (self.history[0], False)

    def _store_history_impl(self, data, path="./history.csv"):
        """Store history to csv file."""
        # convert to pd dataframe
        df = pd.DataFrame(data)
        # move 'job_id' to the first column
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('job_id')))
        df = df.reindex(columns=cols)
        # check if 'time' exists
        if 'time' in df.columns:
            df = df.drop(columns=['time'])
        if 'has_error' in df.columns:
            df = df.drop(columns=['has_error'])
        # write to csv
        df.to_csv(path, index=False)

    def store_history(self, path="./history.csv"):
        # get enhanced report in dp-estimation mode
        if self.search_algo == "dp_estimation":
            metric_name = self.tuner_cfg['metric_cfg']['name']
            if self.additional_metric_key:
                _history = []
                for cfg in self.history:
                    if (
                        "sharding_overlap" not in cfg.keys()
                        or cfg["sharding_overlap"] is None
                    ) and cfg["error_info"] is None:
                        _history.append(copy.deepcopy(cfg))
                _history.sort(
                    key=lambda x: x[self.additional_metric_key]
                    if x[self.additional_metric_key] is not None
                    else float('-inf'),
                    reverse=True,
                )
                self._store_history_impl(
                    data=_history, path=path.split('.csv')[0] + '_enhanced.csv'
                )

        """Store history to csv file."""
        self.store_path = path
        self._store_history_impl(data=self.history, path=path)

    def load_history(self, path="./history.csv") -> Tuple[list, bool]:
        """Load history from csv file."""
        err = False
        if self.store_path is None:
            self.store_path = path
        if not os.path.exists(self.store_path):
            err = True
        else:
            with open(self.store_path, "r") as f:
                reader = csv.reader(f)
                self.history = list(reader)
        return (self.history, err)

    def clean_history(self) -> None:
        """Clean history."""
        self.history = []
