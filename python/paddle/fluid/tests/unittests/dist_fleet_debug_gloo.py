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

<<<<<<< HEAD
import logging

# import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import (
    fleet,
)
=======
from __future__ import print_function
import os
import time
import numpy as np
import logging
import paddle
import paddle.fluid as fluid
#import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)
<<<<<<< HEAD
# role = role_maker.GeneralRoleMaker(
# init_timeout_seconds=100,
# run_timeout_seconds=100,
# http_ip_port="127.0.0.1:26001")

# role = role_maker.PaddleCloudRoleMaker(http_ip_port="127.0.0.1:26001")

# role = role_maker.GeneralRoleMaker(path="./tmp4")
=======
#role = role_maker.GeneralRoleMaker(
#init_timeout_seconds=100,
#run_timeout_seconds=100,
#http_ip_port="127.0.0.1:26001")

#role = role_maker.PaddleCloudRoleMaker(http_ip_port="127.0.0.1:26001")

#role = role_maker.GeneralRoleMaker(path="./tmp4")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
logger.info("Begin")
res = [0, 0]

logger.info(res)

role = role_maker.PaddleCloudRoleMaker(path="./tmp4")

fleet.init(role)
print("init wancheng")  #
<<<<<<< HEAD
# if fleet.is_worker():
=======
#if fleet.is_worker():
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#    import time
#    time.sleep(3)

a = [5]
b = [2]
res = [0]
if fleet.worker_index() == 0:
    role._all_reduce(role._node_type_comm, a)
elif fleet.worker_index() == 1:
    role._all_reduce(role._node_type_comm, b)

<<<<<<< HEAD
# logger.info(res)
# print("res ", res)

# role._barrier_all()
=======
#logger.info(res)
#print("res ", res)

#role._barrier_all()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
