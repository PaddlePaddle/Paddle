#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Fleet Utils."""
"""distributed operations"""
"""basic collective operations in python"""
"""remote file system"""


# __all__ = ['UtilBase']
class UtilBase(object):
    def __init__(self, role_maker, fleet_obj):
        self.role_maker = roke_maker
        self.fleet_obj = fleet_obj

    def set_file_system(self, fs_client):
        self.fs_client = fs_client

    def broadcast(self):
        pass

    def all_gather(self):
        pass

    def all_reduce(self):
        pass

    def reduce_scatter(self):
        pass

    def reduce(self):
        pass

    def get_file_shard(self, files):
        pass

    def feed_gen(self, batch_size, feed_vars_dims, feeded_vars_filelist):
        pass

    def save_program(program, output_dir):
        pass

    def load_program(input_dir):
        pass

    def load_var():
        pass

    def save_var():
        pass

    def print_on_rank(self):
        pass

    def wait_server_ready(endpoints):
        """
        Wait until parameter servers are ready, use connext_ex to detect
        port readiness.

        Args:
        endpoints (list): endpoints string list, like:
        ["127.0.0.1:8080", "127.0.0.1:8081"]

        Examples:
            .. code-block:: python

               wait_server_ready(["127.0.0.1:8080", "127.0.0.1:8081"])
    """
        assert not isinstance(endpoints, string_types)
        while True:
            all_ok = True
            not_ready_endpoints = []
            for ep in endpoints:
                ip_port = ep.split(":")
                with closing(
                        socket.socket(socket.AF_INET,
                                      socket.SOCK_STREAM)) as sock:
                    sock.settimeout(2)
                    result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                    if result != 0:
                        all_ok = False
                        not_ready_endpoints.append(ep)
            if not all_ok:
                sys.stderr.write("server not ready, wait 3 sec to retry...\n")
                sys.stderr.write("not ready endpoints:" + str(
                    not_ready_endpoints) + "\n")
                sys.stderr.flush()
                time.sleep(3)
            else:
                break
