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


def wait_pods_running(label_selector, num):
    print "label selector: %s, num: %s" % (label_selector, num)
    while True:
        pod_list = fetch_pods_info(label_selector)
        running_pod_list = filter(lambda x: x[0] == "Running", pod_list)
        print "running pod list: ", running_pod_list
        if len(running_pod_list) == int(num):
            return [item[1] for item in running_pod_list]
        print "sleep for 10 seconds..."
        time.sleep(10)


def fetch_pserver_ips():
    label_selector = "paddle-job=%s-pserver" % PADDLE_JOB_NAME
    pod_list = fetch_pods_info(label_selector)
    pserver_ips = [item[1] for item in pod_list]
    return ",".join(pserver_ips)


def fetch_trainer_id():
    label_selector = "paddle-job=%s-trainer" % PADDLE_JOB_NAME
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
    elif command == "wait_pods_running":
        wait_pods_running(sys.argv[2], sys.argv[3])
