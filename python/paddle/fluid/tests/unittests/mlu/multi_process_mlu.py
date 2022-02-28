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

import os
import sys
import time
import paddle.fluid as fluid


def train(prefix):
    selected_mlus = os.getenv("FLAGS_selected_mlus")
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
    worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
    worker_endpoints = worker_endpoints_env
    trainers_num = len(worker_endpoints.split(','))

    name = "selected_mlus:{} worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}"\
        .format(selected_mlus, worker_endpoints, trainers_num, current_endpoint,trainer_id)

    print(name)
    with open("multi_process_{}.check_{}.log".format(prefix, trainer_id),
              "w") as f:
        f.write(name)


def train_abort(prefix):
    selected_mlus = os.getenv("FLAGS_selected_mlus")
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
    worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
    worker_endpoints = worker_endpoints_env
    trainers_num = len(worker_endpoints.split(','))

    if trainer_id == 0:
        try:
            # train abort 
            exit(1)
        except SystemExit:
            name = "abort>>> selected_mlus:{} worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}"\
                .format(selected_mlus, worker_endpoints, trainers_num, current_endpoint,trainer_id)
            print(name)
            with open(
                    "multi_process_{}.check_{}.log".format(prefix, trainer_id),
                    "w") as f:
                f.write(name)
            raise
    else:
        # sleep 30s to make sure paddle.distributed.launch will terminate this process
        time.sleep(30)
        name = "selected_mlus:{} worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}"\
            .format(selected_mlus, worker_endpoints, trainers_num, current_endpoint,trainer_id)

        print(name)
        with open("multi_process_{}.check_{}.log".format(prefix, trainer_id),
                  "w") as f:
            f.write(name)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[2] == "abort":
        prefix = sys.argv[1]
        train_abort(prefix)
    else:
        prefix = sys.argv[1]
        train(prefix)
