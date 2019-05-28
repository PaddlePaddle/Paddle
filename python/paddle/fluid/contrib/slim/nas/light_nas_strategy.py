# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from ..core.strategy import Strategy
from ..graph import GraphWrapper
from .controller_server import ControllerServer
from .search_agent import SearchAgent
from ....executor import Executor
import re
import logging
import functools
import socket
import fcntl

__all__ = ['LightNASStrategy']

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class LightNASStrategy(Strategy):
    """
    Light-NAS search strategy.
    """

    def __init__(self,
                 controller=None,
                 end_epoch=1000,
                 target_flops=629145600,
                 retrain_epoch=1,
                 metric_name='top1_acc',
                 server_ip=None,
                 server_port=0,
                 is_server=False,
                 search_steps=None):
        """
        """
        self.start_epoch = 0
        self.end_epoch = end_epoch
        self._max_flops = target_flops
        self._metric_name = metric_name
        self._controller = controller
        self._retrain_epoch = 0
        self._server_ip = server_ip
        self._server_port = server_port
        self._is_server = is_server
        self._retrain_epoch = retrain_epoch
        self._search_steps = search_steps

        if self._server_ip is None:
            self._server_ip = self._get_host_ip()

    def _get_host_ip(self):

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        finally:
            s.close()

        return ip

    def on_compression_begin(self, context):
        self._current_tokens = context.search_space.init_tokens()
        constrain_func = functools.partial(
            self._constrain_func, context=context)
        self._controller.reset(context.search_space.range_table(),
                               self._current_tokens, None)

        # create controller server
        if self._is_server:
            open("./slim_LightNASStrategy_controller_server.socket",
                 'a').close()
            socket_file = open(
                "./slim_LightNASStrategy_controller_server.socket", 'r+')
            fcntl.flock(socket_file, fcntl.LOCK_EX)
            tid = socket_file.readline()
            if tid == '':
                _logger.info("start controller server...")
                self._server = ControllerServer(
                    controller=self._controller,
                    address=(self._server_ip, self._server_port),
                    max_client_num=100,
                    search_steps=self._search_steps)
                tid = self._server.start()
                self._server_port = self._server.port()
                socket_file.write(tid)
                _logger.info("started controller server...")
            fcntl.flock(socket_file, fcntl.LOCK_UN)
            socket_file.close()
        _logger.info("self._server_ip: {}; self._server_port: {}".format(
            self._server_ip, self._server_port))
        # create client
        self._search_agent = SearchAgent(self._server_ip, self._server_port)

    def _constrain_func(self, tokens, context=None):
        """Check whether the tokens meet constraint."""
        _, _, test_prog, _, _, _, _ = context.search_space.create_net(tokens)
        flops = GraphWrapper(test_prog).flops()
        if flops <= self._max_flops:
            return True
        else:
            return False

    def on_epoch_begin(self, context):

        _logger.info("light nas strategy on_epoch_begin")
        if context.epoch_id >= self.start_epoch and context.epoch_id <= self.end_epoch and (
                self._retrain_epoch == 0 or
            (context.epoch_id - self.start_epoch) % self._retrain_epoch == 0):

            startup_p, train_p, test_p, _, _, train_reader, test_reader = context.search_space.create_net(
                self._current_tokens)
            context.eval_graph.program = test_p
            context.train_reader = train_reader
            context.eval_reader = test_reader

            exe = Executor(context.place)
            exe.run(startup_p)

            context.optimize_graph.program = train_p
            context.optimize_graph.compile()

            context.skip_training = (self._retrain_epoch == 0)

    def on_epoch_end(self, context):
        if context.epoch_id >= self.start_epoch and context.epoch_id < self.end_epoch and (
                self._retrain_epoch == 0 or
            (context.epoch_id - self.start_epoch) % self._retrain_epoch == 0):

            self._current_reward = context.eval_results[self._metric_name][-1]
            flops = context.eval_graph.flops()
            self._current_reward = self._current_reward if flops <= self._max_flops else 0
            self._current_tokens = self._search_agent.update(
                self._current_tokens, self._current_reward)
