# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from .prune_strategy import PruneStrategy
import re
import logging

__all__ = ['LightNASStrategy']

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class LightNASStrategy(object):
    """
    Light-NAS search strategy.
    """

    def __init__(self,
                 controller=None,
                 start_epoch=0,
                 end_epoch=0,
                 target_flops=100000,
                 delta=1000,
                 metric_name='top1_acc',
                 pruned_params='conv.*_weights',
                 server_ip=None,
                 server_port=None,
                 is_server=False):
        """
        """
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self._max_flops = target_flops + delta
        self._min_flops = target_flops - delta
        self.controller = controller
        self._retrain_epoch = 0
        self.server_ip = server_ip
        self.server_port = server_port
        self.is_server = is_server

    def on_compression_begin(self, context):
        self._current_tokens = context.search_space.init_tokens()
        constrain_func = functools.partial(
            self._constrain_func, context=context)
        self._controller.reset(
            range_table, constrain_func, init_tokens=self._current_tokens)

        # create controller server
        if self.is_server:
            socket_file = open(
                "~/slim_LightNASStrategy_controller_server.socket", 'w')
            fcntl.flock(socket_file, fcntl.LOCK_EX)
            pid = socket_file.readline()
            if pid == '':
                pid, server = self._create_controller_server(
                    self._controller, self._server_ip, self._server_port)
                server.start()
                socket_file.write(pid)
            fcntl.flock(socket_file, fcntl.LOCK_UN)
            socket_file.close()

        # create client
        self._search_client = self._create_search_client()

    def _constrain_func(self, tokens, context=None):
        """Check whether the tokens meet constraint."""
        program = context.search_space.create_net(tokens)
        flops = GraphWrapper(program).flops()
        if flops >= self._min_flops and flops <= self._max_flops:
            return True
        else:
            return False

    def on_epoch_begin(self, context):
        if context.epoch_id >= self.start_epoch and context.epoch_id <= self.end_epoch and (
                self._retrain_epoch == 0 or
            (context.epoch_id - self.start_epoch) % self._retrain_epoch == 0):
            self._current_token = self._search_client.next_tokens()

            train_program = context.search_space.create_train_net(
                self._current_token)
            eval_program = context.search_space.create_eval_net(
                self._current_token)
            context.train_graph.program = program
            context.eval_graph.program = program
            context.optimize_graph = None

    def on_epoch_end(self, context):
        if context.epoch_id >= self.start_epoch and context.epoch_id < self.end_epoch and (
                self._retrain_epoch == 0 or
            (context.epoch_id - self.start_epoch) % self._retrain_epoch == 0):
            self._current_reward = context.eval_results[-1]
            self._search_client.update(self._current_token,
                                       self._current_reward)
