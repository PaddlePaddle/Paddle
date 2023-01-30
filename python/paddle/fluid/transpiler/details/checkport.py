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

import sys
import time
import socket
from contextlib import closing
<<<<<<< HEAD
=======
from six import string_types
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


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
<<<<<<< HEAD
    assert not isinstance(endpoints, str)
=======
    assert not isinstance(endpoints, string_types)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    while True:
        all_ok = True
        not_ready_endpoints = []
        for ep in endpoints:
            ip_port = ep.split(":")
<<<<<<< HEAD
            with closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as sock:
=======
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as sock:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                sock.settimeout(2)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if hasattr(socket, 'SO_REUSEPORT'):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

                result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                if result != 0:
                    all_ok = False
                    not_ready_endpoints.append(ep)
        if not all_ok:
            sys.stderr.write("server not ready, wait 3 sec to retry...\n")
<<<<<<< HEAD
            sys.stderr.write(
                "not ready endpoints:" + str(not_ready_endpoints) + "\n"
            )
=======
            sys.stderr.write("not ready endpoints:" + str(not_ready_endpoints) +
                             "\n")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            sys.stderr.flush()
            time.sleep(3)
        else:
            break
