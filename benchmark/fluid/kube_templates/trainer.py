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

trainer = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {
        "name": "jobname-pserver"
    },
    "spec": {
        "parallelism": 4,
        "completions": 4,
        "template": {
            "metadata": {
                "labels": {
                    "paddle-job": "jobname"
                }
            },
            "spec": {
                "hostNetwork": True,
                "imagePullSecrets": [{
                    "name": "job-registry-secret"
                }],
                "restartPolicy": "Never",
                "containers": [{
                    "name": "trainer",
                    "image": "",
                    "imagePullPolicy": "Always",
                    # to let container set rlimit
                    "securityContext": {
                        "privileged": True
                        # TODO(wuyi): use below specific cap instead of privileged,
                        # using privileged will cause all GPU device are visible
                        # in the container.
                        # "capabilities": {
                        #     "add": ["SYS_RESOURCE"]
                        # }
                    },
                    "ports": [{
                        "name": "jobport-1",
                        "containerPort": 1
                    }],
                    "env": [],
                    "command": ["paddle_k8s", "start_trainer", "v2"],
                    "resources": {
                        "requests": {
                            "memory": "10Gi",
                            "cpu": "4",
                        },
                        "limits": {
                            "memory": "10Gi",
                            "cpu": "4",
                        }
                    }
                }]
            }
        }
    }
}
