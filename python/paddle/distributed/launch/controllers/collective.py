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

from .controller import Controller, ControleMode
from ..context.device import DeviceType

import json


class CollectiveController(Controller):
    @classmethod
    def enable(cls, ctx):
        # collective is the default mode
        if ctx:
            ctx.logger.debug("{} enabled".format(cls.__name__))
            ctx.args.run_mode = ControleMode.COLLECTIVE
            return True
        else:
            return False

    def build_pod(self):
        if (
            self.ctx.args.master is None
            and self.ctx.args.start_port
            and self.ctx.args.ips
        ):
            self._build_pod_with_args()
        else:
            self._build_pod_with_master()

    def _build_pod_with_args(self):
        self.pod.replicas = self.pod_replicas()

        start_port = int(self.ctx.args.start_port)
        ips = self.ctx.args.ips.split(',')

        job_endpoints = [
            f"{h}:{p+start_port}" for h in ips for p in range(self.pod.replicas)
        ]

        self.ctx.logger.debug("job endpoints: {}".format(job_endpoints))

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
                "PADDLE_GLOBAL_SIZE": "{}".format(len(job_endpoints)),
                "PADDLE_LOCAL_SIZE": "{}".format(self.pod.replicas),
                "PADDLE_GLOBAL_RANK": "{}".format(i + rank_offset),
                "PADDLE_LOCAL_RANK": "{}".format(i),
                "PADDLE_NNODES": "{}".format(len(ips)),
                # compatible env
                "PADDLE_TRAINER_ENDPOINTS": ",".join(job_endpoints),
                "PADDLE_CURRENT_ENDPOINT": job_endpoints[i + rank_offset],
                "PADDLE_TRAINER_ID": "{}".format(i + rank_offset),
                "PADDLE_TRAINERS_NUM": "{}".format(len(job_endpoints)),
                "PADDLE_RANK_IN_NODE": str(i),
            }
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

    def _build_pod_with_master(self):
        self.pod.replicas = self.pod_replicas()

        # rank will be reset when restart
        self.pod.rank = int(self.ctx.args.rank)

        port = self.ctx.node.get_free_port()

        # compatible
        endpoints = [
            "{}:{}".format(self.ctx.node.ip, p)
            for p in self.ctx.node.get_free_ports(self.pod.replicas)
        ]

        data = json.dumps(
            {
                'name': self.pod.name,
                'rank': self.pod.rank,
                'replicas': self.pod.replicas,
                'dtype': self.ctx.node.device.dtype,
                'candidate': '{}:{}'.format(self.ctx.node.ip, port),
                'endpoints': ",".join(endpoints),
            }
        )

        peer_list, rank = self.master.sync_peers(
            '/{}/info'.format(self.job.id),
            self.pod.name,
            data,
            self.job.replicas,
            self.pod.rank,
        )
        self.pod.rank = rank

        if len(peer_list) < 1:
            return False

        peer_list = [json.loads(i) for i in peer_list]

        self.ctx.logger.debug("sync peers done {}".format(peer_list))
        self.save_pod_log(peer_list)

        global_size = sum([i['replicas'] for i in peer_list])
        rank_offset = sum([i['replicas'] for i in peer_list[:rank]])
        '''
        The new designed collective need nothing but a master endpoint
        '''
        collective_master = peer_list[0]['candidate']

        job_endpoints = [i['endpoints'] for i in peer_list]

        self.pod.reset()
        selected_dev_key = self.ctx.node.device.get_selected_device_key()
        selected_dev_list = self.ctx.node.device.get_selected_devices(
            self.ctx.args.devices
        )
        for i in range(self.pod.replicas):
            e = {
                "PADDLE_MASTER": collective_master,
                "PADDLE_GLOBAL_SIZE": "{}".format(global_size),
                "PADDLE_LOCAL_SIZE": "{}".format(self.pod.replicas),
                "PADDLE_GLOBAL_RANK": "{}".format(i + rank_offset),
                "PADDLE_LOCAL_RANK": "{}".format(i),
                "PADDLE_NNODES": "{}".format(self.job.replicas),
                # compatible env
                "PADDLE_TRAINER_ENDPOINTS": ",".join(job_endpoints),
                "PADDLE_CURRENT_ENDPOINT": endpoints[i],
                "PADDLE_TRAINER_ID": "{}".format(i + rank_offset),
                "PADDLE_TRAINERS_NUM": "{}".format(global_size),
                "PADDLE_RANK_IN_NODE": str(i),
            }
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
            ctx.logger.debug("{} enabled".format(cls.__name__))
            ctx.args.run_mode = ControleMode.COLLECTIVE
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
                self.ctx.logger.warning("peer not ready {}".format(self.job))
                break

            self.ctx.logger.debug("Run {}".format(self.job))

            if not self.build_pod():
                continue

            self.master.set_status(self.ctx.status.RUNNING)

            self.deploy_pod()

            if self.watch():
                break

        self.ctx.logger.debug("Job done {}".format(self.job))
