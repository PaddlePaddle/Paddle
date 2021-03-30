# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.contrib.utils import HDFSClient
import os
import time


def check_all_trainers_ready(ready_path, epoch):
    trainer_num = fleet.worker_num()
    trainer_id = fleet.worker_index()

    hadoop_home = os.getenv("HADOOP_HOME")
    configs = {
        "fs.default.name": os.getenv("FS_NAME"),
        "hadoop.job.ugi": os.getenv("FS_UGI")
    }

    node_ready = "ready.{}.{}.done".format(epoch, trainer_id)

    with open(node_ready, "w") as node:
        node.write("")

    client = HDFSClient(hadoop_home, configs)
    if not client.is_dir(ready_path):
        client.makedirs(ready_path)
    client.upload(
        hdfs_path=ready_path,
        local_path=node_ready,
        overwrite=True,
        retry_times=0)

    print("PUT {} ON HDFS {} OK".format(node_ready, ready_path))

    while True:
        ready_num = len(client.ls(ready_path))
        print("have {} trainers need to be ready".format(trainer_num - ready_num
                                                         % trainer_num))
        if ready_num % trainer_num == 0:
            break
        time.sleep(10)
        ready_num = len(client.ls(ready_path))

    print("All trainers are ready, continue training")
