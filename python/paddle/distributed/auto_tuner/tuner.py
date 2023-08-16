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

        search_algo = tuner_cfg.get("search_algo", "grid")

        if search_algo == "grid":
            from .search import GridSearch

            tuner_cfg["candidates"] = default_candidates(tuner_cfg)
            self.algo = GridSearch(tuner_cfg)
        elif search_algo == "gbs":
            from .search import GBSSearch

            tuner_cfg["candidates"] = gbs_default_candidates(tuner_cfg)
            self.algo = GBSSearch(tuner_cfg)
        else:
            raise NotImplementedError()

        self.history_cfgs = []

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
