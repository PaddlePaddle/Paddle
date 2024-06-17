# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import os
import random
import signal
import socket
import subprocess
import threading
import time
import traceback

from paddle.distributed.fleet import cloud_utils, launch_utils
from paddle.distributed.utils.log_utils import get_logger

from ...backup_env import getenv_or_backup

logger = get_logger("INFO", "ELASTIC")

ELASTIC_EXIT_CODE = 101
ELASTIC_AUTO_PARALLEL_EXIT_CODE = 102

# wait for timeout, unit: seconds
ELASTIC_TIMEOUT = 2 * 60

# keepalived ttl, unit: seconds
ELASTIC_TTL = 60


# 1: Fault tolerance, 2: Elastic
class ElasticLevel:
    FAULT_TOLERANCE = 1
    ELASTIC = 2


class ElasticStatus:
    COMPLETED = "completed"
    ERROR = "error"
    HOLD = "hold"
    RESTART = "restart"
    EXIT = "exit"


class LauncherInterface:
    def __init__(self, args):
        self.args = args
        self.procs = []

    def _terminate_procs(self):
        # try to terminate process by group, this happened in multiprocess scenario in user process
        if os.name != 'nt':
            for p in self.procs:
                if p.proc.poll() is None:
                    os.killpg(os.getpgid(p.proc.pid), signal.SIGTERM)
                    if p.log_fn:
                        p.log_fn.close()
                    logger.info(f"terminate process group gid:{p.proc.pid}")

            time.sleep(1)
        for p in self.procs:
            if p.proc.poll() is None:
                p.proc.terminate()
                if p.log_fn:
                    p.log_fn.close()
                logger.info(f"terminate process id:{p.proc.pid}")

        for step in range(0, 50):
            alive = False
            for p in self.procs:
                if p.proc.poll() is None:  # not terminate
                    os.kill(p.proc.pid, signal.SIGKILL)
                    alive = True

            if not alive:
                logger.info("terminated all the procs")
                return True

            time.sleep(1)
        return False

    def _check_procs(self):
        alive = False
        result = None
        for p in self.procs:
            ret = p.proc.poll()
            if ret is None:
                alive = True
            elif ret != 0:
                if ret == ELASTIC_AUTO_PARALLEL_EXIT_CODE:
                    logger.info("return form elastic auto parallel re-launch")
                    return ret
                logger.error("ABORT!!! ABORT!!! ABORT!!!")
                logger.error(
                    f"ERROR rank {p.rank} error with exit code {ret}, check log for detail."
                )
                result = ret
        if not alive and result is None:
            return 0
        else:
            return result

    def launch(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def watch(self):
        raise NotImplementedError


class ElasticManager:
    def __init__(self, args, etcd_client):
        self.args = args
        server = args.elastic_server or os.getenv('PADDLE_ELASTIC_SERVER')
        name = args.job_id or os.getenv('PADDLE_ELASTIC_JOB_ID')
        self.min_np, self.max_np = self._parse_np(args.np)
        host = args.host or os.getenv('POD_IP')
        scale = args.scale or int(os.getenv('PADDLE_ELASTIC_SCALE', 0))
        force = args.force or os.getenv('PADDLE_ELASTIC_FORCE')

        self.host = host if host else self._get_host()

        (
            self.device_mode,
            self.devices_per_proc,
        ) = launch_utils.get_device_proc_info(args)

        self.elastic_timeout = int(
            os.getenv('PADDLE_ELASTIC_TIMEOUT', ELASTIC_TIMEOUT)
        )
        elastic_ttl = int(os.getenv('PADDLE_ELASTIC_TTL', ELASTIC_TTL))

        self.start_port = None
        if cloud_utils.use_paddlecloud():
            self.trainers = os.getenv('PADDLE_TRAINERS', '')
            self.np = len(self.trainers.split(","))
            self.start_port = int(os.getenv("PADDLE_PORT", "6170"))
            self.dist_endpoints = os.getenv('DISTRIBUTED_TRAINER_ENDPOINTS', '')
            trainer_endpoints = getenv_or_backup('PADDLE_TRAINER_ENDPOINTS', '')
            self.trainer_endpoints_list = trainer_endpoints.split(",")
        else:
            self.trainers = args.ips or os.getenv('PADDLE_TRAINERS', '')
            node_ips = self.trainers.split(",")
            self.np = len(node_ips)
            self.start_port = int(os.getenv("FLAGS_START_PORT", "6170"))
            self.dist_endpoints = self._host_to_endpoints(
                node_ips, self.devices_per_proc, self.start_port
            )
            self.trainer_endpoints_list = [
                "%s:%d" % (ip, self.start_port) for ip in node_ips
            ]

        self.curr_host = "%s:%d" % (self.host, self.start_port)
        logger.info(f'start job with np={self.np}')
        logger.info(
            f"trainers={self.trainers}, trainer_endpoints_list={self.trainer_endpoints_list}"
        )

        # auto correct the value of elastic_level
        # 1: Fault tolerant, 2: Elastic
        self.elastic_level = int(
            os.getenv(
                'PADDLE_ELASTIC_FAULT_TOLERANC_LEVEL',
                ElasticLevel.FAULT_TOLERANCE,
            )
        )
        if self.min_np == self.max_np or (self.min_np > 0 and self.max_np == 0):
            self.elastic_level = ElasticLevel.FAULT_TOLERANCE
            logger.info('start job with ElasticLevel.FAULT_TOLERANCE')
        if self.min_np > 0 and self.max_np > self.min_np:
            self.elastic_level = ElasticLevel.ELASTIC
            logger.info('start job with ElasticLevel.ELASTIC')

        # compatible with kubernetes service discovery
        if (
            not server
            and os.getenv('PADDLE_ELASTIC_ETCD_SERVICE_HOST')
            and os.getenv('PADDLE_ELASTIC_ETCD_SERVICE_PORT')
        ):
            server = '{}:{}'.format(
                os.getenv('PADDLE_ELASTIC_ETCD_SERVICE_HOST'),
                os.getenv('PADDLE_ELASTIC_ETCD_SERVICE_PORT'),
            )

        logger.debug(f'init with server {server} host {host}')

        self.hosts = []
        self.stopped = False

        self.sigint = 0
        self.need_sync = False

        self.elastic_startup_time = None

        if not server or ':' not in server or not name or not self.np:
            logger.info(
                f'Elastic is not enabled with server {server} name {name} and np {self.np}'
            )
            self.enable = False
            return
        else:
            self.enable = True

        self.etcd = etcd_client

        # etcd data
        self.prefix = "/paddle/" + name
        self.node_prefix = self.prefix + '/nodes'
        self.np_path = self.prefix + '/np'
        self.endpoints_path = self.prefix + '/endpoints'

        node_tag = ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(6)
        )
        self.host_path = f'{self.node_prefix}/{node_tag}{time.time()}'
        '''
        0 group mode, be aware of healthy status of other workers
        1 decouple mode, check own status only
        '''
        self.etcd.put(self.prefix, b'0')

        # register callback
        def host_call_back(event):
            self.hosts = [
                i[0].decode() for i in self.etcd.get_prefix(self.node_prefix)
            ]
            self.hosts = list(set(self.hosts)) if self.hosts else self.hosts
            logger.info(
                f"host_call_back curr_host={self.curr_host}, hosts:{self.hosts}"
            )
            self.need_sync = True
            self.elastic_startup_time = None

        host_watch = self.etcd.add_watch_prefix_callback(
            self.node_prefix, host_call_back
        )
        host_lease = self.etcd.lease(elastic_ttl)

        # register etcd lease heartbeat
        def lease_heartbeat():
            while True:
                try:
                    host_lease.refresh()

                    hosts = [
                        i[0].decode()
                        for i in self.etcd.get_prefix(self.node_prefix)
                    ]
                    hosts = list(set(hosts)) if hosts else hosts
                    logger.info(
                        f"[lease_heartbeat] curr_host={self.curr_host}, hosts={hosts}"
                    )
                    if self.curr_host not in hosts:
                        logger.info(
                            f"[lease_heartbeat] register host={self.curr_host}"
                        )
                        self.etcd.put(
                            self.host_path,
                            self.curr_host.encode('latin-1'),
                            lease=host_lease,
                        )
                except Exception as e:
                    logger.error(
                        f"[lease_heartbeat] internal error:{e} {traceback.format_exc()}"
                    )
                    break
                time.sleep(elastic_ttl / 3)

        keepalived_thread = threading.Thread(
            name='lease_heartbeat', target=lease_heartbeat, daemon=True
        )
        keepalived_thread.start()

        self.etcd.put(
            self.host_path, self.curr_host.encode('latin-1'), lease=host_lease
        )

        # endpoints handle DISTRIBUTED_TRAINER_ENDPOINTS and PADDLE_TRAINERS
        self.etcd.put(
            self.endpoints_path,
            f'{self.dist_endpoints}|{self.trainers}'.encode('latin-1'),
        )

        def endpoints_call_back(event):
            if not self.dist_endpoints:
                return
            value = self.etcd.get(self.endpoints_path)[0]
            edps = value.decode() if value is not None else ''
            self.dist_endpoints, self.trainers = edps.split('|')
            logger.info(
                f"set DISTRIBUTED_TRAINER_ENDPOINTS {self.dist_endpoints} "
            )
            logger.info(f"set PADDLE_TRAINERS {self.trainers} ")

        endpoints_watch = self.etcd.add_watch_callback(
            self.endpoints_path, endpoints_call_back
        )

        self.watches = [host_watch, endpoints_watch]
        self.launcher = None

    def _host_to_endpoints(
        self, ip_port_list: list, devices_per_proc: list, start_port: int = 6170
    ) -> str:
        endpoint_list = []
        for ip_port in ip_port_list:
            endpoints = ip_port.split(":")
            if len(endpoints) == 2:
                ip = endpoints[0]
                port = int(endpoints[1])
            else:
                ip = endpoints
                port = start_port

            ports = list(range(port, port + len(devices_per_proc)))
            endpoint_list.extend(["%s:%d" % (ip, port) for port in ports])

        dist_endpoints = ','.join(endpoint_list)
        return dist_endpoints

    def exit(self, completed=False):
        logger.info(f'manager exist completed {completed}')

        if self.launcher:
            self.launcher.stop()

        if not self.enable:
            return

        if completed:
            self.etcd.put(self.prefix, b'1')

        for watch in self.watches:
            self.etcd.cancel_watch(watch)
        self.etcd.delete(self.host_path)

        hosts = list(self.etcd.get_prefix(self.node_prefix))
        if len(hosts) == 0:
            self.etcd.delete_prefix(self.prefix)

    def pre_hook(self):
        if not self.args.elastic_pre_hook:
            logger.info("skip pre_hook")
            return
        logger.info("execute pre_hook...")
        current_env = copy.copy(os.environ.copy())
        out, err = subprocess.Popen(
            self.args.elastic_pre_hook,
            env=current_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        ).communicate()
        if err:
            logger.warning("pre_hook exec failed")
        else:
            logger.info(f"pre_hook exec result: {out.decode('utf-8').strip()}")

    def _parse_np(self, np: str):
        """
        np format is "MIN" or "MIN:MAX"
        """
        np_str = np or os.getenv('PADDLE_ELASTIC_NP', "0")
        np_dict = np_str.split(":")
        min_np = max_np = 0
        if len(np_dict) == 1:
            # Fault tolerant
            min_np = int(np_dict[0])
            min_np = 1 if min_np <= 0 else min_np
            max_np = 1
        elif len(np_dict) == 2:
            # Elastic
            min_np = int(np_dict[0])
            max_np = int(np_dict[1])
            min_np = 1 if min_np <= 0 else min_np
            max_np = min_np if min_np > max_np else max_np
        else:
            raise ValueError(
                f'the np={np} needs to be in "MIN" or "MIN:MAX" format'
            )

        return min_np, max_np

    def _get_host(self):
        try:
            return socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        except:
            return '127.0.0.1'

    def _completed(self):
        if not self.enable:
            return True

        return int(self.etcd.get(self.prefix)[0]) == 1

    def _match(self, host_list: list = None):
        if host_list:
            self.hosts = host_list
        else:
            self.hosts = [
                i[0].decode() for i in self.etcd.get_prefix(self.node_prefix)
            ]
        self.hosts = list(set(self.hosts)) if self.hosts else self.hosts

        if self.elastic_level == ElasticLevel.FAULT_TOLERANCE:
            if len(self.hosts) == self.np:
                return True
            else:
                return False

        if self.elastic_level == ElasticLevel.ELASTIC:
            hosts_num = len(self.hosts)
            if hosts_num == self.np:
                return True

            if not self.elastic_startup_time:
                self.elastic_startup_time = time.time()
            if hosts_num == self.max_np:
                self.elastic_startup_time = None
                return True
            elif hosts_num >= self.min_np and hosts_num < self.max_np:
                interval_time = time.time() - self.elastic_startup_time
                if interval_time <= self.elastic_timeout:
                    logger.info(
                        f"wait for timeout, you can set value by PADDLE_ELASTIC_TIMEOUT, \
                        hosts_num={hosts_num}, min_np={self.min_np}, \
                        interval_time={interval_time}, elastic_timeout={self.elastic_timeout}"
                    )
                    return False
                return True
            else:
                self.elastic_startup_time = None
                return False

        return False

    def _update_endpoint(self, endpoints, hosts):
        self.etcd.put(
            self.endpoints_path,
            f'{endpoints}|{hosts}'.encode('latin-1'),
        )

    def _update_fault_tolerance(self):
        rank = int(os.getenv('PADDLE_TRAINER_ID', -1))
        logger.debug(
            f"self.curr_host={self.curr_host}, self.dist_endpoints={self.dist_endpoints}"
        )
        if self.curr_host in self.dist_endpoints:
            os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = self.dist_endpoints
            os.environ['PADDLE_TRAINERS'] = self.trainers
            logger.info(
                f"update env DISTRIBUTED_TRAINER_ENDPOINTS {self.dist_endpoints} "
            )
            logger.info(f"update env PADDLE_TRAINERS {self.trainers} ")
            return

        # fault tolerance
        idx = self.hosts.index(self.curr_host)

        # swap if self.host not in the right position
        if rank >= 0:
            self.hosts[idx] = self.hosts[rank]
            self.hosts[rank] = self.curr_host
        else:
            os.environ['PADDLE_TRAINER_ID'] = f'{idx}'
        hosts = ','.join([host_port.split(":")[0] for host_port in self.hosts])
        self.args.ips = hosts
        os.environ['PADDLE_TRAINERS'] = hosts

    def _update_elastic_scale_out(self):
        host_endpoints = copy.deepcopy(self.trainer_endpoints_list)
        logger.info(
            f"elastic scale out, from {len(self.hosts)} to {self.np}, hosts={self.hosts}, host_endpoints={host_endpoints}"
        )

        for curr_host_port in self.hosts:
            if curr_host_port not in host_endpoints:
                host_endpoints.append(curr_host_port)

        os.environ['PADDLE_TRAINER_ID'] = str(
            host_endpoints.index(self.curr_host)
        )
        hosts = ','.join(
            [host_port.split(":")[0] for host_port in host_endpoints]
        )
        self.args.ips = hosts
        os.environ['PADDLE_TRAINERS'] = hosts
        self.np = len(host_endpoints)
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = ','.join(host_endpoints)
        os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = self.dist_endpoints
        self.trainer_endpoints_list = host_endpoints

    def _update_elastic_scale_in(self):
        host_endpoints = copy.deepcopy(self.trainer_endpoints_list)
        logger.info(
            f"elastic scale in, from {self.np} to {len(self.hosts)}, hosts={self.hosts}, host_endpoints={host_endpoints}"
        )

        # If scale in node from the first of the rank list, you need to minimize the movement of the rank
        # eg:
        #   the source trainers is:10.10.10.0,10.10.10.1,10.10.10.2,10.10.10.3
        #   10.10.10.0 is removed
        #   the new trainers is:10.10.10.3,10.10.10.1,10.10.10.2
        #   In this case, the rank of 10.10.10.1 and 10.10.10.2 remains unchanged, while the rank of 10.10.10.3 is set to rank0
        endpoints_dict = {}
        unsorted_endpoints = []
        for id, host_port in enumerate(self.hosts):
            idx = host_endpoints.index(host_port)
            if idx <= len(self.hosts) - 1 and not endpoints_dict.get(idx):
                endpoints_dict[idx] = host_port
            else:
                unsorted_endpoints.append(host_port)

        idle_index = 0
        sorted_endpoints = []
        for idx in range(len(self.hosts)):
            if not endpoints_dict.get(idx) and len(unsorted_endpoints) > 0:
                endpoints_dict[idx] = unsorted_endpoints[idle_index]
                idle_index += 1

            sorted_endpoints.append(endpoints_dict.get(idx))

        logger.info(f"elastic scale in, sorted_endpoints={sorted_endpoints}")
        self.trainer_endpoints_list = sorted_endpoints

        ip_list = [ip_port.split(":")[0] for ip_port in sorted_endpoints]
        hosts = ','.join(ip_list)
        new_endpoints = self._host_to_endpoints(
            sorted_endpoints, self.devices_per_proc
        )

        self.args.ips = hosts
        os.environ['PADDLE_TRAINER_ID'] = str(
            sorted_endpoints.index(self.curr_host)
        )
        os.environ['PADDLE_TRAINERS'] = hosts
        self.np = len(sorted_endpoints)
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = ','.join(sorted_endpoints)
        os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = new_endpoints
        self._update_endpoint(new_endpoints, hosts)

    def _update_hosts(self):
        assert len(self.hosts) != 0, 'hosts empty'
        if self.elastic_level == ElasticLevel.FAULT_TOLERANCE:
            self._update_fault_tolerance()
        else:
            # elastic
            if len(self.hosts) == self.np:
                logger.info(f"elastic startup, hosts={self.hosts}")
                self._update_fault_tolerance()

            elif len(self.hosts) > self.np:
                # scale out
                self._update_elastic_scale_out()
            else:
                # scale in
                self._update_elastic_scale_in()

    def wait(self):
        if not self.enable:
            return

        idx = 1
        while not self.stopped:
            if self._match():
                logger.info(f'ready with hosts {self.hosts}')
                self._update_hosts()
                return
            logger.info(f'not ready for np {self.np} with hosts {self.hosts}')
            idx += 1
            time.sleep(2)
        return

    def run(self, launcher):
        if self.stopped:
            return

        self.launcher = launcher(self.args)
        self.launcher.launch()

    def watch(self):
        if self.need_sync:
            self.need_sync = False

        while not self.stopped:
            ret = self.launcher.watch()
            logger.debug(f"launcher.watch():{ret}")

            if ret is not None:  # self terminated
                logger.info(f'job exit with code {ret}')
                if ret == ELASTIC_AUTO_PARALLEL_EXIT_CODE:
                    logger.info('job re-launch for auto parallel')
                    self.launcher.stop()
                    return ElasticStatus.HOLD

                # process is completed if ret >= 0 or error else
                completed = True if ret == 0 else False
                self.exit(completed=completed)
                if completed:
                    return ElasticStatus.COMPLETED
                if self.elastic_level == ElasticLevel.FAULT_TOLERANCE:
                    return ElasticStatus.RESTART
                else:
                    return ElasticStatus.ERROR

            if not self._completed() and (not self._match() or self.need_sync):
                self.launcher.stop()
                return ElasticStatus.HOLD

            time.sleep(2)

        if self.launcher:
            self.launcher.stop()

        return ElasticStatus.EXIT

    def signal_handler(self, sigint, frame):
        if self.enable:
            self.exit()
        self.sigint = sigint
        self.stopped = True
