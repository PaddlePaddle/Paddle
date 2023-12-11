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
from typing import Tuple

import pandas as pd


class HistoryRecorder:
    # NOTE increase extenable ablitity
    def __init__(self) -> None:
        self.history = []
        self.store_path = None

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

    def get_best(self, metric, direction, mode=None) -> Tuple[dict, bool]:
        self.sort_metric(direction=direction, metric_name=metric)
        if len(self.history) == 0:
            return (self.history[0], True)
        if mode == "SFT" or mode == "LoRA":
            best_cfg = self.history[0]
            if (
                isinstance(best_cfg["max_mem_usage"], str)
                or best_cfg["time"] == -1
            ):
                return (best_cfg, True)
            first_few = 1
            for cfg in self.history:
                if (
                    not isinstance(cfg["max_mem_usage"], str)
                    and cfg["max_mem_usage"] < best_cfg["max_mem_usage"]
                    and cfg["time"] != -1
                ):
                    best_cfg = cfg
                first_few += 1
                if first_few >= 5:
                    break
            return (best_cfg, False)
        if isinstance(self.history[0]["max_mem_usage"], str) or (
            "time" in self.history[0] and self.history[0]["time"] == -1
        ):
            return (self.history[0], True)
        return (self.history[0], False)

    def store_history(self, path="./history.csv"):
        """Store history to csv file."""
        self.store_path = path
        # convert to pd dataframe
        df = pd.DataFrame(self.history)
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
        df.to_csv(self.store_path, index=False)

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
