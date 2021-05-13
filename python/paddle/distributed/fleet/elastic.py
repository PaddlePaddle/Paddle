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

import etcd3
import time
import socket
import os
import six
import logging


class Status:
    READY = 'ready'
    RUNNING = 'running'
    ERROR = 'error'
    COMPLETED = 'completed'


class ElasticManager(object):
    def __init__(self, server, name, np, host=None, scale=0, force=False):

        logging.info('[elastic] init with server {} host {}'.format(server,
                                                                    host))

        srv, port = server.split(':')
        self.etcd = etcd3.client(host=srv, port=port)
        self.host = host if host else self._get_host()
        self.hosts = []

        # etcd data
        self.prefix = "/paddle/" + name
        self.node_prefix = self.prefix + '/nodes/'
        self.np_path = self.prefix + '/np'
        self.host_path = self.node_prefix + self.host

        self.np = np + scale
        '''
        0 group mode, be aware of healthy status of other workers
        1 decouple mode, check own status only
        '''
        self.etcd.put(self.prefix, b'0')

        # host
        # register self host to etcd
        # register watch to reset host after host been deleted
        self.etcd.delete_prefix(self.node_prefix)
        self.etcd.put(self.host_path, six.b(self.host))

        def host_call_back(event):
            if self.etcd.get(self.host_path)[0] == None:
                logging.info('[elastic] register host agin {}'.format(
                    self.host))
                self.etcd.put(self.host_path, six.b(self.host))

        host_watch = self.etcd.add_watch_callback(self.host_path,
                                                  host_call_back)

        # np
        #
        inp = int(self.etcd.get(self.np_path)[0] or 0)
        if scale == 0 and not force:
            assert (
                inp == np,
                "[elastic] np {} is not consistent with np in etcd {}, maybe the job with the same name exited unexpected, try --force=true".
                format(np, inp))
        else:
            assert (inp == np or inp == self.np,
                    "[elastic] np {} scale to {} by {} is not allowed".format(
                        inp, self.np, scale))

        self.etcd.put(self.np_path, six.b("%d" % (self.np)))

        def np_call_back(event):
            gnp = int(self.etcd.get(self.np_path)[0])
            if gnp != self.np:
                logging.info("[elastic] scale np {} to {} ".format(self.np,
                                                                   gnp))
                self.np = gnp

        np_watch = self.etcd.add_watch_callback(self.np_path, np_call_back)

        self.watches = [host_watch, np_watch]

    def exit(self, completed=False):
        logging.info('[elastic] manager exist completed {}'.format(completed))

        if completed:
            self.etcd.put(self.prefix, b'1')

        for watch in self.watches:
            self.etcd.cancel_watch(watch)
        self.etcd.delete(self.host_path)

        hosts = [i for i in self.etcd.get_prefix(self.node_prefix)]
        if len(hosts) == 0:
            self.etcd.delete_prefix(self.prefix)

    def _get_host(self):
        try:
            return socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        except:
            return '127.0.0.1'

    def _completed(self):
        return int(self.etcd.get(self.prefix)[0]) == 1

    def _match(self):
        self.hosts = [
            six.ensure_str(i[0]) for i in self.etcd.get_prefix(self.node_prefix)
        ]
        if len(self.hosts) == self.np:
            return True
        else:
            return False

    def ready(self):
        while True:
            if self._match():
                logging.info('[elastic] ready with hosts {}'.format(self.hosts))
                return True
            logging.info('[elastic] not ready for np {} with hosts {}'.format(
                self.np, self.hosts))
            time.sleep(3)
        return False

    def health(self):
        return self._completed() or self._match()

    def signal_handler(self, sigint, frame):
        self.exit()
        exit(0)
