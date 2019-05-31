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

__all__ = ['SearchAgent']

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class SearchAgent(object):
    """
    Search agent.
    """

    def __init__(self, server_ip=None, server_port=None, key=None):
        """
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._key = key

    def update(self, tokens, reward):
        socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_client.connect((self.server_ip, self.server_port))
        tokens = ",".join([str(token) for token in tokens])
        socket_client.send("{}\t{}\t{}".format(self._key, tokens, reward))
        tokens = socket_client.recv(1024).decode()
        tokens = [int(token) for token in tokens.strip("\n").split(",")]
        return tokens

    def next_tokens(self):
        socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_client.connect((self.server_ip, self.server_port))
        socket_client.send("next_tokens")
        tokens = socket_client.recv(1024).decode()
        tokens = [int(token) for token in tokens.strip("\n").split(",")]
        return tokens
