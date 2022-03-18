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

from .job.container import Container
from .job.pod import Pod
from .job.job import Job
from . import plugins

#__all__ = [Container, Pod, Job]
'''
Paddle distribution training entry ``python -m paddle.distributed.run``.

Help

# for arg usage and explanation, try the following command
# python -m paddle.distributed.run -h

Collective Mode

Case 1: 1 node

use all visible devices
# python -m paddle.distributed.run train.py

use specified devices
# python -m paddle.distributed.run --devices=0,1,2,3 train.py

Case 2: multi-node, auto detect ip/port

# python -m paddle.distributed.run --np 2 train.py
# auto print following command
# python -m paddle.distributed.run --master 10.0.0.1:13538 --np 2 demo.py
# then copy and paste above command to other nodes

Case 3: multi-node, specified master/rendezvous server

# python -m paddle.distributed.run --np 2 --master 10.0.0.1:2379 train.py
# the master ip must be one of the node and the port must available

Parameter Server Mode

Case 1.1: 1 node, 1 ps, 1 trainer

# python -m paddle.distributed.run --mode ps train.py
# python -m paddle.distributed.run --server_num=1 --trainer_num=1 train.py

Case 1.2: 1 node, 2 ps, 2 trainer

# python -m paddle.distributed.run --server_num=2 --trainer_num=2 train.py

Case 2: 2 node, 2 ps, 2 trainer per node

# python -m paddle.distributed.run --server_num=2 --trainer_num=2 --np 2 train.py
# auto print following command
# python -m paddle.distributed.run --master 10.0.0.1:13538 --server_num=2 --trainer_num=2 --np 2 train.py
# then copy and paste above command to other nodes

Case 3: multi-node, specified master/rendezvous server

# python -m paddle.distributed.run --master 10.0.0.1:13538 --server_num=2 --trainer_num=2 --np 2 train.py
# the master ip must be one of the node and the port must available

Case 4: specified servers and trainers in each node

python -m paddle.distributed.run --servers 127.0.0.1:8900,127.0.0.1:8901 --trainers 127.0.0.1:8902,127.0.0.1:8903 train.py


Elastic Mode

# run following command in 3 node to run immediately, or in 2 node to run after elastic_timeout
# python -m paddle.distributed.run --master etcd://10.0.0.1:2379 --np 2:3 train.py

# once the peer number changes between 2:3, the strategy holds

'''
