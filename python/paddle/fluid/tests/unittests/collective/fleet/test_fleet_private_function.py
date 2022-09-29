# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import socket
import threading


class TestFleetPrivateFunction(unittest.TestCase):

    def test_wait_port(self):

        def init_server(port):
            import time
            time.sleep(5)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            sock.listen(10)
            while True:
                c, addr = sock.accept()
                c.send("0")
                c.close()
                break

        thr = threading.Thread(target=init_server, args=(9292, ))
        thr.start()

        import paddle.distributed.fleet as fleet
        ep = ["127.0.0.1:9292"]
        fleet.base.private_helper_function.wait_server_ready(ep)

        thr.join()


if __name__ == "__main__":
    unittest.main()
