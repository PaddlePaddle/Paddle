#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import ctypes
import os

__lib__ = None


def get_c_lib():
    global __lib__
    if __lib__ is None:
        path = os.path.join(os.path.dirname(__file__), "libpaddle_master.so")
        __lib__ = ctypes.cdll.LoadLibrary(path)
    return __lib__


class client(object):
    """
    client is a client to the master server.
    """

    def __init__(self, etcd_endpoints, timeout_sec, buf_size=0):
        self.c = get_c_lib().paddle_new_etcd_master_client(
            etcd_endpoints, timeout_sec, buf_size)

    def request_save_model(self, trainer_id, block_ms):
        """request to save model

        Conventionally the 0-th trainer will save model. But in
        distributed training, any trainer could be killed. This
        function asks the master server if the trainer should proceed
        with saving model.

        :param trainer_id: trainer id.
        :param block_ms: number of millisecond that other save model
        will be blocked if this save model request succeeded.

        Returns:
            int: 1 if the save the model request is approved, 0 if
            does the request is rejected because other trainer is
            saving the model, -1 if error happened.

        """
        return get_c_lib().paddle_request_save_model(self.c, trainer_id,
                                                     block_ms)

    def release(self):
        get_c_lib().paddle_release_master_client(self.c)
        self.c = None

    def set_dataset(self, paths):
        holder_type = ctypes.c_char_p * len(paths)
        holder = holder_type()
        for idx, path in enumerate(paths):
            c_ptr = ctypes.c_char_p(path)
            holder[idx] = c_ptr
        get_c_lib().paddle_set_dataset(self.c, holder, len(paths))

    def next_record(self):
        """gets next record for training

        Returns:
            string: the record.
            int: error code, 0 if successful, < 0 otherwise.
        """
        p = ctypes.c_char_p()
        ret = ctypes.pointer(p)
        size = get_c_lib().paddle_next_record(self.c, ret)
        if size < 0:
            # Error
            return None, size

        if size == 0:
            # Empty record
            return "", 0

        record = ret.contents.value[:size]
        # Memory created from C should be freed.
        get_c_lib().mem_free(ret.contents)
        return record, 0

    def paddle_start_get_records(self, pass_id):
        get_c_lib().paddle_start_get_records(self.c, pass_id)
