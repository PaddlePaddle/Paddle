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

import json
import os

from ..context.device import DeviceType
from .controller import Controller, ControllerMode


class CollectiveController(Controller):
    def __init__(self, ctx):
        self._tuner_run_mode = None  # 'tuner_only', 'run_only', 'tuner_and_run'
        super().__init__(ctx)

    @classmethod
    def enable(cls, ctx):
        # collective is the default mode
        if ctx:
            ctx.logger.debug(f"{cls.__name__} enabled")
            ctx.args.run_mode = ControllerMode.COLLECTIVE
            return True
        else:
            return False

    def build_pod(self):
        skip_run = self._build_pod_with_tuner()
        if skip_run:
            return
        if (
            self.ctx.args.master is None
            and self.ctx.args.start_port
            and self.ctx.args.ips
        ):
            return self._build_pod_with_args()
        else:
            if self.ctx.args.auto_parallel_config is None:
                skip_run = True
            # only when skip_run is False, should not reset pod
            return self._build_pod_with_master(skip_run)

    def _build_pod_with_tuner(self):
        auto_parallel_config = self.ctx.args.auto_parallel_config
        if auto_parallel_config is not None:
            if not os.path.exists(auto_parallel_config):
                self.ctx.logger.warning("auto_parallel_conf not exists!")
            if not auto_parallel_config.endswith(".json"):
                self.ctx.logger.warning(
                    "auto_parallel_config should be a json format file!"
                )

            with open(auto_parallel_config, 'r') as robj:
                auto_parallel_data = json.loads(robj.read())
                self._tuner_run_mode = auto_parallel_data.get(
                    "tuner_run_mode", 'tuner_and_run'
                )

            self.ctx.logger.info(f"tuner_run_mode is: {self._tuner_run_mode}")
            endpoint = f"127.0.0.1:{self.ctx.node.get_free_port()}"
            pod_replicas = self.pod_replicas()
            if self._tuner_run_mode in ['tuner_only', 'tuner_and_run']:
                e = {
                    "PADDLE_AUTO_PARALLEL_CONFIG": self.ctx.args.auto_parallel_config,
                    "PADDLE_TRAINERS_NUM": "1",
                    "PADDLE_TRAINER_ENDPOINTS": endpoint,
                    "PADDLE_TRAINER_ID": "0",
                    "PADDLE_CURRENT_ENDPOINT": endpoint,
                    "FLAGS_selected_gpus": "0",
                    "PADDLE_AUTO_PARALLEL_STAGE": "tuner",
                    "PADDLE_GLOBAL_SIZE": f"{pod_replicas * int(self.ctx.args.nnodes)}",
                    "PADDLE_LOCAL_SIZE": f"{pod_replicas}",
                }
                log_file = "tuner.log"
                self.add_container(envs=e, log_file=log_file, is_init=True)

                if self._tuner_run_mode == 'tuner_only':
                    return True
        return False

    def _build_pod_with_args(self):
        self.pod.replicas = self.pod_replicas()

        start_port = int(self.ctx.args.start_port)
        ips = self.ctx.args.ips.split(',')

        job_endpoints = [
            f"{h}:{p + start_port}"
            for h in ips
            for p in range(self.pod.replicas)
        ]

        self.ctx.logger.debug(f"job endpoints: {job_endpoints}")

        self.ctx.logger.warning(
            f"master is set by args, it will be overwritten by {job_endpoints[0]}."
        )
        # this is necessary for tcp store to work when endpoints cannot be passed to sub processes.
        self.ctx.args.master = job_endpoints[0]

        rank_offset = (
            ips.index(self.ctx.node.ip) * self.pod.replicas
            if self.ctx.node.ip in ips
            else 0
        )

        self.save_pod_log(job_endpoints)

        selected_dev_key = self.ctx.node.device.get_selected_device_key()
        selected_dev_list = self.ctx.node.device.get_selected_devices(
            self.ctx.args.devices
        )

        for i in range(self.pod.replicas):
            e = {
                "PADDLE_MASTER": self.ctx.args.master,
                "PADDLE_GLOBAL_SIZE": f"{len(job_endpoints)}",
                "PADDLE_LOCAL_SIZE": f"{self.pod.replicas}",
                "PADDLE_GLOBAL_RANK": f"{i + rank_offset}",
                "PADDLE_LOCAL_RANK": f"{i}",
                "PADDLE_NNODES": f"{len(ips)}",
                # compatible env
                "PADDLE_CURRENT_ENDPOINT": job_endpoints[i + rank_offset],
                "PADDLE_TRAINER_ID": f"{i + rank_offset}",
                "PADDLE_TRAINERS_NUM": f"{len(job_endpoints)}",
                "PADDLE_RANK_IN_NODE": str(i),
                "PADDLE_AUTO_CLUSTER": str(self.ctx.args.auto_cluster_config),
            }
            e.update({"PADDLE_TRAINER_ENDPOINTS": ",".join(job_endpoints)})

            if self._tuner_run_mode is not None:
                e.update(
                    {
                        "PADDLE_AUTO_PARALLEL_CONFIG": self.ctx.args.auto_parallel_config,
                        "PADDLE_AUTO_PARALLEL_STAGE": "run",
                    }
                )
            if len(selected_dev_list) > 0:
                if self.ctx.node.device.dtype == DeviceType.CUSTOM_DEVICE:
                    e.update(self.ctx.node.device.get_custom_device_envs())
                if self.pod.replicas == 1:
                    e.update({selected_dev_key: ",".join(selected_dev_list)})
                else:
                    e.update({selected_dev_key: selected_dev_list[i]})
            else:
                e.update({'PADDLE_DISTRI_BACKEND': 'gloo'})

            log_file = f"workerlog.{i}"
            self.add_container(envs=e, log_file=log_file)

        return True

    def _build_pod_with_master(self, reset_pod=True):
        self.pod.replicas = self.pod_replicas()

        # rank will be reset when restart
        self.pod.rank = int(self.ctx.args.rank)

        port = self.ctx.node.get_free_port()

        # compatible
        endpoints = [
            f"{self.ctx.node.ip}:{p}"
            for p in self.ctx.node.get_free_ports(
                self.pod.replicas, self.pod.rank
            )
        ]

        data = json.dumps(
            {
                'name': self.pod.name,
                'rank': self.pod.rank,
                'replicas': self.pod.replicas,
                'dtype': self.ctx.node.device.dtype,
                'candidate': f'{self.ctx.node.ip}:{port}',
                'endpoints': ",".join(endpoints),
            }
        )

        peer_list, rank = self.master.sync_peers(
            f'/{self.job.id}/info',
            self.pod.name,
            data,
            self.job.replicas,
            self.pod.rank,
        )
        self.pod.rank = rank

        if len(peer_list) < 1:
            return False

        peer_list = [json.loads(i) for i in peer_list]

        self.ctx.logger.debug(f"sync peers done {peer_list}")
        self.save_pod_log(peer_list)

        global_size = sum([i['replicas'] for i in peer_list])
        rank_offset = sum([i['replicas'] for i in peer_list[:rank]])
        '''
        The new designed collective need nothing but a master endpoint
        '''
        collective_master = peer_list[0]['candidate']

        # get collective master ip
        collective_master_ip = collective_master.split(':')[0].strip()
        os.environ["COLLECTIVE_MASTER_IP"] = collective_master_ip

        job_endpoints = [i['endpoints'] for i in peer_list]

        if reset_pod:
            self.pod.reset()
        selected_dev_key = self.ctx.node.device.get_selected_device_key()
        selected_dev_list = self.ctx.node.device.get_selected_devices(
            self.ctx.args.devices
        )
        for i in range(self.pod.replicas):
            e = {
                "PADDLE_MASTER": collective_master,
                "PADDLE_GLOBAL_SIZE": f"{global_size}",
                "PADDLE_LOCAL_SIZE": f"{self.pod.replicas}",
                "PADDLE_GLOBAL_RANK": f"{i + rank_offset}",
                "PADDLE_LOCAL_RANK": f"{i}",
                "PADDLE_NNODES": f"{self.job.replicas}",
                # compatible env
                "PADDLE_CURRENT_ENDPOINT": endpoints[i],
                "PADDLE_TRAINER_ID": f"{i + rank_offset}",
                "PADDLE_TRAINERS_NUM": f"{global_size}",
                "PADDLE_RANK_IN_NODE": str(i),
                "PADDLE_AUTO_CLUSTER": str(self.ctx.args.auto_cluster_config),
            }
            e.update({"PADDLE_TRAINER_ENDPOINTS": ",".join(job_endpoints)})

            if self._tuner_run_mode is not None:
                e.update(
                    {
                        "PADDLE_AUTO_PARALLEL_CONFIG": self.ctx.args.auto_parallel_config,
                        "PADDLE_AUTO_PARALLEL_STAGE": "run",
                    }
                )
            if len(selected_dev_list) > 0:
                if self.ctx.node.device.dtype == DeviceType.CUSTOM_DEVICE:
                    e.update(self.ctx.node.device.get_custom_device_envs())
                if self.pod.replicas == 1:
                    e.update({selected_dev_key: ",".join(selected_dev_list)})
                else:
                    e.update({selected_dev_key: selected_dev_list[i]})
            else:
                e.update({'PADDLE_DISTRI_BACKEND': 'gloo'})

            # log_file = "{}.{}.{}.log".format(self.job.id, self.pod.name, i)
            log_file = f"workerlog.{i}"
            self.add_container(envs=e, log_file=log_file)

        return True


class CollectiveElasticController(CollectiveController):
    @classmethod
    def enable(cls, ctx):
        if ctx.args.master and ctx.args.master.startswith("etcd://"):
            ctx.logger.debug(f"{cls.__name__} enabled")
            ctx.args.run_mode = ControllerMode.COLLECTIVE
            return True
        else:
            return False

    def register(self):
        if self.job.id == 'default':
            self.ctx.logger.warning(
                'Using default job name may cause conflict, add --job_id in args'
            )

        self.master.register_heartbeat(self.job.id, self.pod.name)

    def run(self):
        timeout = int(self.ctx.args.elastic_timeout)
        timeout = timeout if self.job.elastic else timeout * 10
        self.register()

        while self.pod.restart <= self.ctx.args.max_restart:
            self.build_job()

            self.ctx.logger.info("Waiting peer ready...")

            ok, replicas = self.master.wait_peer_ready(
                self.job.replicas_min, self.job.replicas_max, timeout
            )
            if ok:
                self.job.replicas = replicas
            else:
                self.ctx.logger.warning(f"peer not ready {self.job}")
                if self.ctx.is_auto_tuner_mode():
                    self.ctx.logger.info(
                        "Failed to start peer, auto tuner exit."
                    )
                    import sys

                    sys.exit(-1)
                break

            self.ctx.logger.debug(f"Run {self.job}")

            if not self.build_pod():
                continue

            self.master.set_status(self.ctx.status.RUNNING)

            self.deploy_pod()

            if self.watch():
                break

        self.ctx.logger.debug(f"Job done {self.job}")
