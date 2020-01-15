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

import logging
import socket
from threading import Thread
from ....log_helper import get_logger

__all__ = ['ControllerServer']

_logger = get_logger(
    __name__,
    logging.INFO,
    fmt='ControllerServer-%(asctime)s-%(levelname)s: %(message)s')


class ControllerServer(object):
    """
    The controller wrapper with a socket server to handle the request of search agentt.
    """

    def __init__(self,
                 controller=None,
                 address=('', 0),
                 max_client_num=100,
                 search_steps=None,
                 key=None):
        """
        Args:
            controller(slim.searcher.Controller): The controller used to generate tokens.
            address(tuple): The address of current server binding with format (ip, port). Default: ('', 0).
                            which means setting ip automatically
            max_client_num(int): The maximum number of clients connecting to current server simultaneously. Default: 100.
            search_steps(int): The total steps of searching. None means never stopping. Default: None 
        """
        self._controller = controller
        self._address = address
        self._max_client_num = max_client_num
        self._search_steps = search_steps
        self._closed = False
        self._port = address[1]
        self._ip = address[0]
        self._key = key

    def start(self):
        self._socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket_server.bind(self._address)
        self._socket_server.listen(self._max_client_num)
        self._port = self._socket_server.getsockname()[1]
        self._ip = self._socket_server.getsockname()[0]
        _logger.info("listen on: [{}:{}]".format(self._ip, self._port))
        thread = Thread(target=self.run)
        thread.start()
        return str(thread)

    def close(self):
        """Close the server."""
        self._closed = True

    def port(self):
        """Get the port."""
        return self._port

    def ip(self):
        """Get the ip."""
        return self._ip

    def run(self):
        _logger.info("Controller Server run...")
        while ((self._search_steps is None) or
               (self._controller._iter <
                (self._search_steps))) and not self._closed:
            conn, addr = self._socket_server.accept()
            message = conn.recv(1024).decode()
            if message.strip("\n") == "next_tokens":
                tokens = self._controller.next_tokens()
                tokens = ",".join([str(token) for token in tokens])
                conn.send(tokens.encode())
            else:
                _logger.info("recv message from {}: [{}]".format(addr, message))
                messages = message.strip('\n').split("\t")
                if (len(messages) < 3) or (messages[0] != self._key):
                    _logger.info("recv noise from {}: [{}]".format(addr,
                                                                   message))
                    continue
                tokens = messages[1]
                reward = messages[2]
                tokens = [int(token) for token in tokens.split(",")]
                self._controller.update(tokens, float(reward))
                tokens = self._controller.next_tokens()
                tokens = ",".join([str(token) for token in tokens])
                conn.send(tokens.encode())
                _logger.info("send message to {}: [{}]".format(addr, tokens))
            conn.close()
        self._socket_server.close()
        _logger.info("server closed!")
