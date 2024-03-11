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

import logging
import socket
import time

from paddle.distributed.launch import plugins

from .args_envs import env_args_mapping, fetch_envs, parse_args
from .node import Node
from .status import Status


class Context:
    def __init__(self, enable_plugin=True):
        self.args, self.unknown_args = parse_args()
        self.envs = fetch_envs()

        self.set_env_in_args()

        self.node = Node()
        self.status = Status()

        self.update_args()
        self.logger = self.get_logger()

        # design for event queue, later
        self.events = []

        if enable_plugin:
            self._enable_plugin()
        self.max_time_per_task = -1
        self.run_best = False

        self._ip = None
        self._port = None

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

    def is_auto_tuner_mode(self):
        if self.args.auto_tuner_json:
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
        # forbid the child logger pass on to its parent
        logger.propagate = False
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

    def has_set(self, k):
        # check if the args has been set by user
        default_values = {
            "master": None,
            "rank": -1,
            "sort_ip": False,
            "legacy": False,
            "log_level": "INFO",
            "enable_gpu_log": True,
            "nnodes": "-1",
            "nproc_per_node": None,
            "log_dir": "log",
            "run_mode": None,
            "job_id": "default",
            "devices": None,
            "gpus": None,
            "npus": None,
            "xpus": None,
            "ips": None,
            "auto_parallel_config": None,
            "auto_tuner_json": None,
            "servers": "",
            "trainers": "",
            "trainer_num": None,
            "server_num": None,
            "gloo_port": 6767,
            "with_gloo": "1",
            "max_restart": 3,
            "elastic_level": -1,
            "elastic_timeout": 30,
            "host": None,
        }

        if k in default_values and getattr(self.args, k) != default_values[k]:
            return True

        return False

    def set_env_in_args(self):
        # set args by env
        # if user defined, use user value
        for k, v in env_args_mapping.items():
            attr, attr_type = v
            if k in self.envs:
                if attr in self.args and self.has_set(attr):
                    continue
                else:
                    print(
                        f"LAUNCH WARNNING args {attr} will be overridden by env: {k} value: {self.envs[k]}"
                    )
                    setattr(self.args, attr, attr_type(self.envs[k]))

    def update_by_ips(self):
        try:
            ip_list = []
            with open(self.args.ips, "r") as file:
                lines = file.readlines()
                for line in lines:
                    ip = line.split(" ")[0].strip()
                    ip_list.append(ip)
        except:
            ip_list = self.args.ips.split(",")

        if self.has_set("nnodes"):
            nnodes = int(self.args.nnodes)
            if nnodes < len(ip_list):
                ip_list = ip_list[:nnodes]
                print(
                    f"LAUNCH WARNNING only the first {nnodes} nnodes in ip_list will be retained when nnodes {nnodes} < len(ip_list)"
                )
            elif nnodes > len(ip_list):
                raise ValueError(
                    f"LAUNCH ERROR the nnodes {nnodes} > len(ip_list)"
                )

        self.args.ips = ",".join(ip_list)
        self.args.rank = ip_list.index(self.node.ip)
        self.args.nnodes = str(len(ip_list))

        if self.args.rank == -1:
            print(
                f"LAUNCH WARNNING ip {self.node.ip} is not in ip_list: {ip_list}, launch exited."
            )
            return

        if self.has_set("master"):
            if self._ip and self._ip not in ip_list:
                print(
                    f"LAUNCH WARNNING master {self.args.master} will be reset when master not in ip_list {ip_list}."
                )
            else:
                return
        else:
            master_ip = ip_list[0]
            self._ip = master_ip
            if self.envs.get("PADDLE_PORT", None):
                port = self.envs['PADDLE_PORT']
                self.args.master = f"{master_ip}:{port}"
            if self._port:
                port = self._port
                self.args.master = f"{master_ip}:{port}"
            else:
                # magic port
                port = 6768
                has_connect = False
                if self.node.ip == master_ip:
                    while port < 6778:
                        ret = self.node._get_free_port(port)
                        if ret > 0:
                            server_socket = socket.socket(
                                socket.AF_INET, socket.SOCK_STREAM
                            )
                            server_socket.bind(self.node.ip, port)
                            server_socket.listen()
                            connection_count = 0
                            connections = []

                            while True:
                                connection, address = server_socket.accept()
                                message = connection.recv(1024).decode()
                                if message == "connect master":
                                    connection_count += 1
                                    connections.append([connection])
                                if connection_count == len(ip_list):
                                    break
                            for connection in connections:
                                response = "connect success"
                                connection.send(response.encode())
                            has_connect = True
                        else:
                            port += 1
                    server_socket.close()
                else:
                    while port < 6778:
                        has_connect = False
                        for i in range(5):
                            try:
                                client_socket = socket.socket(
                                    socket.AF_INET, socket.SOCK_STREAM
                                )
                                client_socket.connect(self.node.ip, port)
                                message = "connect master"
                                client_socket.send(message.encode())
                                response = client_socket.recv(1024).decode()
                                if response == "connect success":
                                    has_connect = True
                                    break
                            except:
                                time.sleep(2)
                        if has_connect:
                            break
                        else:
                            port += 1
                assert has_connect
                self.args.master = f"{master_ip}:{port}"
                self._ip = master_ip
                self._port = port

    def update_args(self):
        # support master: <ip>:<port>, <ip>, :<port>
        if self.has_set(self.args.master):
            if "etcd" not in self.args.master:
                if ":" in self.args.master:
                    if len(self.args.master.split(":")) == 2:
                        ip, port = self.args.master.split(":")
                        if ip:
                            self._ip = ip
                        self._port = port
                    elif len(self.args.master.split(":")) == 1:
                        value = self.args.master.split(":")[0]
                        if "." in value:
                            self._ip = value
                        else:
                            self._port = value
                    else:
                        raise ValueError(
                            "LAUNCH ERROR master {self.args.master} is invalid."
                        )

        if self.has_set("ips"):
            self.update_by_ips()

        # reset nnodes default value if nnodes not set
        if self.args.nnodes == "-1":
            self.args.nnodes = "1"

        # update master by env
        if "etcd" not in self.args.master:
            if not self._ip:
                if self.envs.get("PADDLE_TRAINERS", None):
                    self._ip = self.envs["PADDLE_TRAINERS"].split(",")[0]
                else:
                    raise ValueError("LAUNCH ERROR the master ip error.")
            if not self._port:
                if self.envs.get("PADDLE_PORT", None):
                    self._port = self.envs.get("PADDLE_PORT")
                else:
                    raise ValueError("LAUNCH ERROR the master port error.")
            self.args.master = f"{self._ip}:{self._port}"
