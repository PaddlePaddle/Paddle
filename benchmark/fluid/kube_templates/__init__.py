# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from pserver import pserver
from trainer import trainer

__all__ = ["pserver", "trainer", "envs"]

envs = [
    # envs that don't need to change
    {
        "name": "GLOG_v",
        "value": "0"
    },
    {
        "name": "GLOG_logtostderr",
        "value": "1"
    },
    {
        "name": "TOPOLOGY",
        "value": ""
    },
    {
        "name": "TRAINER_PACKAGE",
        "value": "/workspace"
    },
    {
        "name": "PADDLE_INIT_NICS",
        "value": "eth2"
    },
    {
        "name": "NAMESPACE",
        "valueFrom": {
            "fieldRef": {
                "fieldPath": "metadata.namespace"
            }
        }
    },
    {
        "name": "POD_IP",
        "valueFrom": {
            "fieldRef": {
                "fieldPath": "status.podIP"
            }
        }
    },
    {
        "name": "PADDLE_CURRENT_IP",
        "valueFrom": {
            "fieldRef": {
                "fieldPath": "status.podIP"
            }
        }
    }
]
