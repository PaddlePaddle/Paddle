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


import logging
import os
from abc import ABC, abstractmethod

from .prune import _PRUNE_HISTORY_FUNC
from .utils import (
    gbs_search_all,
    load_configs_from_csv,
    search_all,
    search_by_dp_estimation,
)

logger = logging.getLogger('auto_tuner')


class SearchAlgo(ABC):
    def __init__(self, tuner_cfg):
        self.tuner_cfg = tuner_cfg
        self.pruned_cfgs = []

    @abstractmethod
    def search_once(self, history_cfgs):
        pass

    def prune(self, tuner_cfg, cur_cfg, history_cfgs, pruned_cfgs):
        for func in _PRUNE_HISTORY_FUNC:
            result = func(tuner_cfg, cur_cfg, history_cfgs, pruned_cfgs)
            if result:
                return True
        return False


class GridSearch(SearchAlgo):
    def __init__(self, tuner_cfg):
        super().__init__(tuner_cfg)
        self.idx = 0
        self.all_tasks = search_all(tuner_cfg)
        need_baseline = self.tuner_cfg.get("need_baseline", False)
        self.baseline = None
        if need_baseline:
            from .utils import memory_sort

            self.all_tasks.sort(key=memory_sort)
        self.previous_cfg = None

    def search_once(self, history_cfgs):
        new_cfg = None
        stop = False
        if history_cfgs:
            if history_cfgs[-1].get("time", -1) > 0:
                if self.baseline is None and self.tuner_cfg.get(
                    "need_baseline", False
                ):
                    from .utils import performance_sort

                    self.baseline = history_cfgs[-1]
                    self.all_tasks[self.idx :] = sorted(
                        self.all_tasks[self.idx : len(self.all_tasks)],
                        key=performance_sort,
                    )
                    if self.tuner_cfg.get("schedule_prior", False):
                        from .utils import sort_by_special

                        self.all_tasks[self.idx :] = sort_by_special(
                            self.all_tasks[self.idx :], self.tuner_cfg
                        )
        while not stop:
            if self.idx < len(self.all_tasks):
                new_cfg = self.all_tasks[self.idx]
                self.idx += 1
                stop = not self.prune(
                    self.tuner_cfg, new_cfg, history_cfgs, self.pruned_cfgs
                )
                self.pruned_cfgs.append(new_cfg)
            else:
                return None

        return new_cfg


class DpEstimationSearch(SearchAlgo):
    def __init__(self, tuner_cfg):
        super().__init__(tuner_cfg)
        self.idx = 0
        if tuner_cfg["candidates"]["dp_degree"] != [1]:
            logger.warning(
                "dp_degree should be [1] in dp estimation search mode. Modify it to [1] automatically."
            )
            tuner_cfg["candidates"]["dp_degree"] = [1]
        self.all_tasks = search_by_dp_estimation(tuner_cfg)
        assert (
            len(self.all_tasks) > 0
        ), "Unable to perform single dp estimation search."

    def search_once(self, history_cfgs):
        new_cfg = None
        stop = False
        while not stop:
            if self.idx < len(self.all_tasks):
                new_cfg = self.all_tasks[self.idx]
                self.idx += 1
                stop = not self.prune(self.tuner_cfg, new_cfg, history_cfgs)
            else:
                return None
        return new_cfg


class GBSSearch(SearchAlgo):
    def __init__(self, tuner_cfg):
        super().__init__(tuner_cfg)
        self.idx = 0
        self.all_tasks = gbs_search_all(tuner_cfg)

    def search_once(self, history_cfgs):
        new_cfg = None
        stop = False
        while not stop:
            if self.idx < len(self.all_tasks):
                new_cfg = self.all_tasks[self.idx]
                self.idx += 1
                glb = new_cfg.get("global_batch_size", None)
                self.tuner_cfg["model_cfg"]["global_batch_size"] = glb
                stop = not self.prune(self.tuner_cfg, new_cfg, history_cfgs)
            else:
                return None
        return new_cfg


class CustomizeSearch(SearchAlgo):
    def __init__(self, tuner_cfg):
        super().__init__(tuner_cfg)
        self.idx = 0
        self.configs_csv = tuner_cfg.get("configs_csv", None)
        assert os.path.exists(
            self.configs_csv
        ), "configs_csv file is necessary in CustomizeSearch mode."
        self.all_tasks = load_configs_from_csv(self.configs_csv)

    def search_once(self, history_cfgs):
        new_cfg = self.all_tasks[self.idx]
        self.idx += 1
        return new_cfg
