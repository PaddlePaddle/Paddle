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

import logging
import socket
from threading import Thread

__all__ = ['ControllerServer']

logging.basicConfig(
    format='ControllerServer-%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class ControllerServer(object):
    """
    """

    def __init__(self,
                 controller=None,
                 address=None,
                 max_client_num=None,
                 search_steps=None):
        """
        """
        self._controller = controller
        self._address = address
        self._max_client_num = max_client_num
        self._search_steps = search_steps
        self._closed = False

    def start(self):
        thread = Thread(target=self.run)
        thread.start()
        return str(thread)

    def close(self):
        self._closed = True

    def run(self):
        _logger.info("Controller Server run...")
        socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_server.bind(self._address)
        socket_server.listen(self._max_client_num)
        _logger.info("listen on: [{}]".format(self._address))
        while ((self._search_steps is None) or
               (self._controller._iter <
                (self._search_steps))) and not self._closed:
            conn, addr = socket_server.accept()
            message = conn.recv(1024).decode()
            _logger.info("recv message from {}: [{}]".format(addr, message))
            tokens, reward = message.strip('\n').split("\t")
            tokens = [int(token) for token in tokens.split(",")]
            self._controller.update(tokens, float(reward))
            tokens = self._controller.next_tokens()
            tokens = ",".join([str(token) for token in tokens])
            conn.send(tokens)
            _logger.info("send message to {}: [{}]".format(addr, tokens))
            conn.close()
        socket_server.close()
        _logger.info("server closed!")
