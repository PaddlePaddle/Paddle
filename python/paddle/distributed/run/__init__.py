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

from .job.container import Container
from .job.pod import Pod
from .job.job import Job
from . import plugins

#__all__ = [Container, Pod, Job]
'''
Paddle distribution training entry ``python -m paddle.distributed.run``.

Case 1: collective, 1 node

use all visible devices
# python -m paddle.distributed.run train.py

use specified devices
# python -m paddle.distributed.run --gpus=0,1,2,3 train.py

Case 2: multi-node collective, auto detect ip/port

# python -m paddle.distributed.run --np 2 train.py
# auto print following command
# python -m paddle.distributed.run --master 10.0.0.1:13538 --np 2 demo.py

Case 3: multi-node collective, specified master/rendezvous server

# python -m paddle.distributed.run --np 2 --master 10.0.0.1:2379 train.py

'''
