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

import subprocess
import shlex
import os
import json
import shutil


class Info(object):

    def __repr__(self):
        return str(self.__dict__)

    def json(self):
        return json.dumps(self.__dict__)

    def dict(self):
        return self.__dict__

    def str(self, keys=None):
        if keys is None:
            keys = self.__dict__.keys()

        if isinstance(keys, str):
            keys = keys.split(',')

        values = [str(self.__dict__.get(k, '')) for k in keys]
        return ",".join(values)


def query_smi(query=None, query_type="gpu", index=None, dtype=None):
    """
    query_type: gpu/compute
    """

    if not has_nvidia_smi():
        return []

    cmd = ["nvidia-smi", "--format=csv,noheader,nounits"]
    if isinstance(query, list) and query_type == "gpu":
        cmd.extend(["--query-gpu={}".format(",".join(query))])
    elif isinstance(query, list) and query_type.startswith("compute"):
        cmd.extend(["--query-compute-apps={}".format(",".join(query))])
    else:
        return

    if isinstance(index, list) and len(index) > 0:
        cmd.extend(["--id={}".format(",".join(index))])
    if not isinstance(dtype, list) or len(dtype) != len(query):
        dtype = [str] * len(query)

    output = subprocess.check_output(cmd, timeout=3)
    lines = output.decode("utf-8").split(os.linesep)
    ret = []
    for line in lines:
        if not line:
            continue
        info = Info()
        for k, v, d in zip(query, line.split(", "), dtype):
            setattr(info, k.replace(".", "_"), d(v))
        ret.append(info)
    return ret


def get_gpu_info(index=None):
    q = "index,uuid,driver_version,name,gpu_serial,display_active,display_mode".split(
        ",")
    d = [int, str, str, str, str, str, str]
    index = index if index is None or isinstance(
        index, list) else str(index).split(",")

    return query_smi(q, index=index, dtype=d)


def get_gpu_util(index=None):
    q = "index,utilization.gpu,memory.total,memory.used,memory.free,timestamp".split(
        ",")
    d = [int, int, int, int, int, str]
    index = index if index is None or isinstance(
        index, list) else str(index).split(",")

    return query_smi(q, index=index, dtype=d)


def get_gpu_process(index=None):
    q = "pid,process_name,gpu_uuid,gpu_name,used_memory".split(",")
    d = [int, str, str, str, int]
    index = index if index is None or isinstance(
        index, list) else str(index).split(",")

    return query_smi(q, index=index, query_type="compute", dtype=d)


def has_nvidia_smi():
    return shutil.which("nvidia-smi")


if __name__ == '__main__':
    print(get_gpu_info(0))
    print(get_gpu_util(0))
    print(get_gpu_process(0))

    u = get_gpu_util()
    for i in u:
        print(i.str())
