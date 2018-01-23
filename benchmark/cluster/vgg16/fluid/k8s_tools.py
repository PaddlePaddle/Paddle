#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

#!/bin/env python
import os
import sys
import time
import socket
from kubernetes import client, config
PADDLE_JOB_NAME = os.getenv("PADDLE_JOB_NAME")
NAMESPACE = os.getenv("NAMESPACE")
PORT = os.getenv("PSERVER_PORT")
if os.getenv("KUBERNETES_SERVICE_HOST", None):
    config.load_incluster_config()
else:
    config.load_kube_config()
v1 = client.CoreV1Api()


def fetch_pods_info(label_selector):
    api_response = v1.list_namespaced_pod(
        namespace=NAMESPACE, pretty=True, label_selector=label_selector)
    pod_list = []
    for item in api_response.items:
        pod_list.append((item.status.phase, item.status.pod_ip))
    return pod_list


def wait_pods_running(label_selector, desired):
    print "label selector: %s, desired: %s" % (label_selector, desired)
    while True:
        count = count_pods_by_phase(label_selector, 'Running')
        # NOTE: pods may be scaled.
        if count >= int(desired):
            break
        print 'current cnt: %d sleep for 5 seconds...' % count
        time.sleep(5)


def count_pods_by_phase(label_selector, phase):
    pod_list = fetch_pods_info(label_selector)
    filtered_pod_list = filter(lambda x: x[0] == phase, pod_list)
    return len(filtered_pod_list)


def fetch_pserver_ips():
    label_selector = "paddle-job-pserver=%s" % PADDLE_JOB_NAME
    pod_list = fetch_pods_info(label_selector)
    pserver_ips = [item[1] for item in pod_list]
    return ",".join(pserver_ips)


def fetch_master_ip():
    label_selector = "paddle-job-master=%s" % PADDLE_JOB_NAME
    pod_list = fetch_pods_info(label_selector)
    master_ips = [item[1] for item in pod_list]
    return master_ips[0]


def fetch_trainer_id():
    label_selector = "paddle-job=%s" % PADDLE_JOB_NAME
    pod_list = fetch_pods_info(label_selector)
    trainer_ips = [item[1] for item in pod_list]
    trainer_ips.sort()
    local_ip = socket.gethostbyname(socket.gethostname())
    for i in xrange(len(trainer_ips)):
        if trainer_ips[i] == local_ip:
            return i
    return None


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "fetch_pserver_ips":
        print fetch_pserver_ips()
    elif command == "fetch_trainer_id":
        print fetch_trainer_id()
    elif command == "fetch_master_ip":
        print fetch_master_ip()
    elif command == "count_pods_by_phase":
        print count_pods_by_phase(sys.argv[2], sys.argv[3])
    elif command == "wait_pods_running":
        wait_pods_running(sys.argv[2], sys.argv[3])
