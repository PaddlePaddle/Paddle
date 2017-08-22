# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import errno

import paddle.v2.master

__all__ = ["save_model", "load_model"]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_model(parameters, folder, name="model.tar"):
    need_request = "KUBERNETES_SERVICE_HOST" in os.environ.keys()

    if need_request:
        # TODO(helin): figure out how MPI trains, since MPI only save
        # model when trainer_id == "0", we can consolidate the logic
        # here.
        trainer_key = "PADDLE_INIT_TRAINER_ID"
        if trainer_key not in os.environ.keys():
            raise Exception('not find ' + trainer_key +
                            ' in environment variable.')
        trainer_id = os.environ.get(trainer_key)
        etcd_key = "ETCD_IP"
        if etcd_key not in os.environ.keys():
            # non-fault-tolerant training
            if trainer_id != "0":
                # only trainer 0 saves model
                return
        else:
            etcd_ip = os.environ.get(etcd_key)
            client = master.client("http://" + etcd_ip + ":2379", 5, 0)
            r = client.request_save_model(trainer_id, 5000)
            if r == 0:
                # do not need to save
                return
            elif r < 0:
                # error
                return
        # save model
        folder = os.path.join(folder, trainer_id)
    path = os.path.join(folder, name)
    mkdir_p(folder)
    with open(path, 'wb') as f:
        parameters.to_tar(f)


def load_model(parameters, path):
    with open(path, 'rb') as f:
        parameters.from_tar(f)
