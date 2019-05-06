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

import os


def train():
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
    worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
    current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
    worker_endpoints = worker_endpoints_env.split(",")
    trainers_num = len(worker_endpoints)

    print("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
              trainer_id:{}"
          .format(worker_endpoints, trainers_num, current_endpoint, trainer_id))


if __name__ == '__main__':
    train()
