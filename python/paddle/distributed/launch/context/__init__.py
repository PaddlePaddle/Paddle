# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.launch import plugins

from .node import Node
from .status import Status
from .args_envs import parse_args, fetch_envs, env_args_mapping

import logging


class Context:
    def __init__(self, enable_plugin=True):
        self.args, self.unknown_args = parse_args()
        self.envs = fetch_envs()

        self.set_env_in_args()

        self.node = Node()
        self.status = Status()

        self.logger = self.get_logger()

        # design for event queue, later
        self.events = []

        if enable_plugin:
            self._enable_plugin()

    def print(self):
        self.logger.info("-----------  Configuration  ----------------------")
        for arg, value in sorted(vars(self.args).items()):
            self.logger.info(f"{arg}: {value}")
        self.logger.info("--------------------------------------------------")

    def is_legacy_mode(self):
        if self.args.legacy:
            return True

        if self.args.master:
            return False

        if len(self.unknown_args) > 0:
            self.logger.warning(
                f"Compatible mode enable with args {self.unknown_args}"
            )
            return True

        return False

    def get_envs(self):
        return self.envs.copy()

    def set_envs(self, env={}):
        env = {k: v for k, v in env.items() if isinstance(v, str)}
        self.envs.update(env)

    def _enable_plugin(self):
        for pl in plugins.enabled_plugins:
            pl(self)

    def get_logger(self, level=logging.INFO):
        logger = logging.getLogger("LAUNCH")
        logger.setLevel(self.args.log_level.upper() or level)
        formatter = logging.Formatter(
            fmt='%(name)s %(levelname)s %(asctime)s %(message)s'
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def continous_log(self) -> bool:
        if self.args.log_level.upper() in ['DEBUG', 'ERROR']:
            return True
        else:
            return False

    def set_env_in_args(self):
        for k, v in env_args_mapping.items():
            if k in self.envs:
                print(
                    f"LAUNCH WARNNING args {v} is override by env {self.envs[k]}"
                )
                setattr(self.args, v, self.envs[k])
