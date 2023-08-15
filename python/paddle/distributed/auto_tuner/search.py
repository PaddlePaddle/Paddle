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


from abc import ABC, abstractmethod

from .prune import _PRUNE_FUNC
from .utils import gbs_search_all, search_all


class SearchAlgo(ABC):
    def __init__(self, tuner_cfg):
        self.tuner_cfg = tuner_cfg

    @abstractmethod
    def search_once(self, history_cfgs):
        pass

    def prune(self, tuner_cfg, cur_cfg, history_cfgs):
        for func in _PRUNE_FUNC:
            result = func(tuner_cfg, cur_cfg, history_cfgs)
            if result:
                return True
        return False


class GridSearch(SearchAlgo):
    def __init__(self, tuner_cfg):
        super().__init__(tuner_cfg)
        self.idx = 0
        self.all_tasks = search_all(tuner_cfg)

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
