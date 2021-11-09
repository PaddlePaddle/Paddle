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

import time
import socket
import os
import six
import logging
import signal
import random
import threading
import traceback
from paddle.distributed.fleet import cloud_utils

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())
logger = logging.getLogger("ELASTIC")

ELASTIC_EXIT_CODE = 101

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


class LauncherInterface(object):
    def __init__(self, args):
        self.args = args
        self.procs = []

    def _terminate_procs(self):
        # try to terminate process by group, this happend in multiprocess senario in user process
        if os.name != 'nt':
            for p in self.procs:
                if p.proc.poll() is None:
                    os.killpg(os.getpgid(p.proc.pid), signal.SIGTERM)
                    if p.log_fn:
                        p.log_fn.close()
                    logger.info("terminate process group gid:{}".format(
                        p.proc.pid))

            time.sleep(1)
        for p in self.procs:
            if p.proc.poll() is None:
                p.proc.terminate()
                if p.log_fn:
                    p.log_fn.close()
                logger.info("terminate process id:{}".format(p.proc.pid))

        for step in range(0, 50):
            alive = False
            for p in self.procs:
                if p.proc.poll() is None:  # not termniate
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
                logger.error("ABORT!!! ABORT!!! ABORT!!!")
                logger.error(
                    "ERROR rank {} error with exit code {}, check log for detail.".
                    format(p.rank, ret))
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


class ElasticManager(object):
    def __init__(self, args, etcd_client):

        self.args = args
        server = args.elastic_server or os.getenv('PADDLE_ELASTIC_SERVER')
        name = args.job_id or os.getenv('PADDLE_ELASTIC_JOB_ID')
        self.min_np, self.max_np = self._parse_np(args.np)
        np = self.min_np
        host = args.host or os.getenv('POD_IP')
        scale = args.scale or int(os.getenv('PADDLE_ELASTIC_SCALE', 0))
        force = args.force or os.getenv('PADDLE_ELASTIC_FORCE')

        start_port = 6170
        if os.environ.get('FLAGS_START_PORT') is not None:
            start_port = int(os.environ.get('FLAGS_START_PORT'))
        if cloud_utils.use_paddlecloud() and self.max_np != 1:
            start_port = int(os.getenv("PADDLE_PORT", ""))

        self.elastic_timeout = int(
            os.getenv('PADDLE_ELASTIC_TIMEOUT', ELASTIC_TIMEOUT))
        elastic_ttl = int(os.getenv('PADDLE_ELASTIC_TTL', ELASTIC_TTL))
        self.endpoints = os.getenv('DISTRIBUTED_TRAINER_ENDPOINTS', '')
        self.trainers = os.getenv('PADDLE_TRAINERS', '')
        self.lastest_endpoints = self.endpoints
        logger.info(
            f"trainers={self.trainers}, lastest_endpoints={self.lastest_endpoints}"
        )

        # auto correct the value of elastic_level
        # 1: Fault tolerant, 2: Elastic
        self.elastic_level = int(
            os.getenv('PADDLE_ELASTIC_FAULT_TOLERANC_LEVEL',
                      ElasticLevel.FAULT_TOLERANCE))
        if self.min_np == self.max_np or \
                (self.min_np > 0 and self.max_np == 0):
            self.elastic_level = ElasticLevel.FAULT_TOLERANCE
        if self.min_np > 0 and self.max_np > self.min_np:
            self.elastic_level = ElasticLevel.ELASTIC

        # compatible with kuberntes service discovery
        if not server and os.getenv(
                'PADDLE_ELASTIC_ETCD_SERVICE_HOST') and os.getenv(
                    'PADDLE_ELASTIC_ETCD_SERVICE_PORT'):
            server = '{}:{}'.format(
                os.getenv('PADDLE_ELASTIC_ETCD_SERVICE_HOST'),
                os.getenv('PADDLE_ELASTIC_ETCD_SERVICE_PORT'))

        logger.debug('init with server {} host {}'.format(server, host))

        self.hosts = []
        self.stopped = False

        self.sigint = 0
        self.need_sync = False

        self.elastic_startup_time = None

        if not server or ':' not in server or not name or not np:
            logger.info(
                'Elastic is not enabled with server {} name {} and np {}'.
                format(server, name, np))
            self.enable = False
            return
        else:
            self.enable = True

        self.etcd = etcd_client
        self.host = host if host else self._get_host()
        self.host_port = "%s:%d" % (self.host, start_port)

        # etcd data
        self.prefix = "/paddle/" + name
        self.node_prefix = self.prefix + '/nodes'
        self.np_path = self.prefix + '/np'
        self.endpoints_path = self.prefix + '/endpoints'

        node_tag = ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(6))
        self.host_path = '{}/{}{}'.format(self.node_prefix, node_tag,
                                          time.time())

        self.np = np + scale
        '''
        0 group mode, be aware of healthy status of other workers
        1 decouple mode, check own status only
        '''
        self.etcd.put(self.prefix, b'0')

        # register callback
        def host_call_back(event):
            self.hosts = [
                six.ensure_str(i[0])
                for i in self.etcd.get_prefix(self.node_prefix)
            ]
            logger.info(
                f"host_call_back curr_host={self.host_port}, hosts:{self.hosts}")
            self.np = len(self.hosts)
            self.need_sync = True
            self.elastic_startup_time = None

        host_watch = self.etcd.add_watch_prefix_callback(self.node_prefix,
                                                         host_call_back)
        host_lease = self.etcd.lease(elastic_ttl)

        # register etcd lease heartbeat
        def lease_heartbeat():
            while True:
                try:
                    host_lease.refresh()

                    hosts = [
                        six.ensure_str(i[0])
                        for i in self.etcd.get_prefix(self.node_prefix)
                    ]
                    logger.info(
                        f"[lease_heartbeat] curr_host={self.host_port}, hosts={hosts}"
                    )
                    if self.host_port not in hosts:
                        logger.info(
                            f"[lease_heartbeat] register host={self.host_port}")
                        self.etcd.put(self.host_path,
                                      six.b(self.host_port),
                                      lease=host_lease)
                except Exception as e:
                    logger.error("[lease_heartbeat] internal error:{} {}".
                                 format(e, traceback.format_exc()))
                    break
                time.sleep(elastic_ttl / 3)

        keepalived_thread = threading.Thread(
            name='lease_heartbeat', target=lease_heartbeat, daemon=True)
        keepalived_thread.start()

        self.etcd.put(self.host_path, six.b(self.host_port), lease=host_lease)

        # endpoints handle DISTRIBUTED_TRAINER_ENDPOINTS and PADDLE_TRAINERS
        self.etcd.put(self.endpoints_path,
                      six.b('{}|{}'.format(self.endpoints, self.trainers)))

        def endpoints_call_back(event):
            if not self.endpoints:
                return
            edps = six.ensure_str(self.etcd.get(self.endpoints_path)[0] or '')
            self.endpoints, self.trainers = edps.split('|')
            logger.info("set DISTRIBUTED_TRAINER_ENDPOINTS {} ".format(
                self.endpoints))
            logger.info("set PADDLE_TRAINERS {} ".format(self.trainers))

        endpoints_watch = self.etcd.add_watch_callback(self.endpoints_path,
                                                       endpoints_call_back)

        self.watches = [host_watch, endpoints_watch]
        self.launcher = None

    def exit(self, completed=False):
        logger.info('manager exist completed {}'.format(completed))

        if self.launcher:
            self.launcher.stop()

        if not self.enable:
            return

        if completed:
            self.etcd.put(self.prefix, b'1')

        for watch in self.watches:
            self.etcd.cancel_watch(watch)
        self.etcd.delete(self.host_path)

        hosts = [i for i in self.etcd.get_prefix(self.node_prefix)]
        if len(hosts) == 0:
            self.etcd.delete_prefix(self.prefix)

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
                f'the np={np} needs to be in "MIN" or "MIN:MAX" format')

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

    def _match(self, host_list: list=None):
        if host_list:
            self.hosts = host_list
        else:
            self.hosts = [
                six.ensure_str(i[0])
                for i in self.etcd.get_prefix(self.node_prefix)
            ]

        if self.elastic_level == ElasticLevel.FAULT_TOLERANCE:
            if len(self.hosts) == self.np:
                return True
            else:
                return False

        if self.elastic_level == ElasticLevel.ELASTIC:
            # FIXME(xym) add freeze status
            hosts_num = len(self.hosts)
            alloc_hosts_num = len(self.endpoints.split(","))
            if hosts_num == alloc_hosts_num:
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

    def _update_endpoint(self, endpoints, hosts):
        self.etcd.put(self.endpoints_path,
                      six.b('{}|{}'.format(endpoints, hosts)))

    def _update_hosts(self):
        assert len(self.hosts) != 0, 'hosts empty'
        rank = int(os.getenv('PADDLE_TRAINER_ID', -1))
        if self.elastic_level == ElasticLevel.FAULT_TOLERANCE:
            self.lastest_endpoints = self.endpoints
            if self.host_port in self.endpoints:
                os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = self.endpoints
                os.environ['PADDLE_TRAINERS'] = self.trainers
                logger.info("update env DISTRIBUTED_TRAINER_ENDPOINTS {} ".
                            format(self.endpoints))
                logger.info("update env PADDLE_TRAINERS {} ".format(
                    self.trainers))
                return

            # fault tolerance 
            idx = self.hosts.index(self.host_port)

            # swap if self.host not in the right position
            if rank >= 0:
                self.hosts[idx] = self.hosts[rank]
                self.hosts[rank] = self.host_port
            else:
                os.environ['PADDLE_TRAINER_ID'] = '{}'.format(idx)
            hosts = ','.join(
                [host_port.split(":")[0] for host_port in self.hosts])
            self.args.ips = hosts
            os.environ['PADDLE_TRAINERS'] = hosts
        else:
            # elastic, scale up/down
            endpoints = self.lastest_endpoints.split(",")
            if len(self.hosts) > len(endpoints):
                # scale up
                logger.info(
                    f"elastic scale up, hosts={self.hosts}, endpoints={endpoints}"
                )

                for curr_host_port in self.hosts:
                    if curr_host_port not in endpoints:
                        endpoints.append(curr_host_port)

                os.environ['PADDLE_TRAINER_ID'] = '{}'.format(
                    endpoints.index(self.host_port))
                host_port_list = ','.join(endpoints)
                hosts = ','.join(
                    [host_port.split(":")[0] for host_port in endpoints])
                self.args.ips = hosts
                os.environ['PADDLE_TRAINERS'] = hosts
                os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = host_port_list
                self.lastest_endpoints = host_port_list
                self._update_endpoint(host_port_list, hosts)
            else:
                # scale down
                logger.info(
                    f"elastic scale down, hosts={self.hosts}, endpoints={endpoints}"
                )

                # If the shrink node is from the first of the rank list, you need to minimize the movement of the rank
                # eg: 
                #   the source trainers is:10.10.10.0,10.10.10.1,10.10.10.2,10.10.10.3
                #   10.10.10.0 is removed
                #   the new trainers is:10.10.10.3,10.10.10.1,10.10.10.2
                #   In this case, the rank of 10.10.10.1 and 10.10.10.2 remains unchanged, while the rank of 10.10.10.3 is set to rank0
                endpoints_dict = dict()
                unsorted_endpoints = []
                for id, host_port in enumerate(self.hosts):
                    idx = endpoints.index(host_port)
                    if idx <= len(self.hosts) - 1 and not endpoints_dict.get(
                            idx):
                        endpoints_dict[idx] = host_port
                    else:
                        unsorted_endpoints.append(host_port)

                idle_index = 0
                sorted_endpoints = []
                for idx in range(len(self.hosts)):
                    if not endpoints_dict.get(idx) and len(
                            unsorted_endpoints) > 0:
                        endpoints_dict[idx] = unsorted_endpoints[idle_index]
                        idle_index += 1

                    sorted_endpoints.append(endpoints_dict.get(idx))

                logger.info(
                    f"elastic scale down, sorted_endpoints={sorted_endpoints}")
                host_port_list = ','.join(sorted_endpoints)
                hosts = ','.join([
                    host_port.split(":")[0] for host_port in sorted_endpoints
                ])
                self.args.ips = hosts
                os.environ['PADDLE_TRAINER_ID'] = '{}'.format(
                    sorted_endpoints.index(self.host_port))
                os.environ['PADDLE_TRAINERS'] = hosts
                os.environ['DISTRIBUTED_TRAINER_ENDPOINTS'] = host_port_list
                self.lastest_endpoints = host_port_list
                self._update_endpoint(host_port_list, hosts)

    def wait(self):
        if not self.enable:
            return

        idx = 1
        while not self.stopped:
            if self._match():
                logger.info('ready with hosts {}'.format(self.hosts))
                self._update_hosts()
                return
            logger.info('not ready for np {} with hosts {}'.format(self.np,
                                                                   self.hosts))
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
                logger.info('job exit with code {}'.format(ret))
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
