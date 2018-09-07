#!/usr/bin/python
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

import requests
import time
import socket
import os
import argparse

# configuration for cluster
API = "/api/v1/namespaces/"
JOBSELECTOR = "labelSelector=job-name="
JOB_PATH = os.getenv("JOB_PATH") + "/" + os.getenv("JOB_NAME")
JOB_PATH_OUTPUT = JOB_PATH + "/output"
JOBNAME = os.getenv("JOB_NAME")
NAMESPACE = os.getenv("JOB_NAMESPACE")
PADDLE_NIC = os.getenv("CONF_PADDLE_NIC")
PADDLE_PORT = os.getenv("CONF_PADDLE_PORT")
PADDLE_PORTS_NUM = os.getenv("CONF_PADDLE_PORTS_NUM")
PADDLE_PORTS_NUM_SPARSE = os.getenv("CONF_PADDLE_PORTS_NUM_SPARSE")
PADDLE_SERVER_NUM = os.getenv("CONF_PADDLE_GRADIENT_NUM")

tokenpath = '/var/run/secrets/kubernetes.io/serviceaccount/token'


def refine_unknown_args(cmd_args):
    '''
    refine unknown parameters to handle some special parameters
    '''
    new_args = []
    for arg in cmd_args:
        if arg.startswith("--") and arg.find("=") != -1:
            equal_pos = arg.find("=")  # find first = pos
            arglist = list(arg)
            arglist[equal_pos] = " "
            arg = "".join(arglist)
            arg = arg.lstrip("-")
            new_args += arg.split(" ")
        elif arg.startswith("--") and arg.find("=") == -1:
            arg = arg.lstrip("-")
            new_args.append(arg)
        else:
            new_args.append(arg)
    return new_args


def isPodAllRunning(podlist):
    '''
    check all pod is running
    '''
    require = len(podlist["items"])
    running = 0
    for pod in podlist["items"]:
        if pod["status"]["phase"] == "Running":
            running += 1
    print "waiting for pods running, require:", require, "running:", running
    if require == running:
        return True
    return False


def getPodList():
    '''
    get all container status of the job
    '''
    apiserver = "https://" + \
        os.getenv("KUBERNETES_SERVICE_HOST") + ":" + \
        os.getenv("KUBERNETES_SERVICE_PORT_HTTPS")

    pod = API + NAMESPACE + "/pods?"
    job = JOBNAME
    if os.path.isfile(tokenpath):
        tokenfile = open(tokenpath, mode='r')
        token = tokenfile.read()
        Bearer = "Bearer " + token
        headers = {"Authorization": Bearer}
        return requests.get(apiserver + pod + JOBSELECTOR + job,
                            headers=headers,
                            verify=False).json()
    else:
        return requests.get(apiserver + pod + JOBSELECTOR + job,
                            verify=False).json()


def getIdMap(podlist):
    '''
    generate tainer_id by ip
    '''
    ips = []
    for pod in podlist["items"]:
        ips.append(pod["status"]["podIP"])
    ips.sort()
    idMap = {}
    for i in range(len(ips)):
        idMap[ips[i]] = i
    return idMap


def startPaddle(idMap={}, train_args_dict=None):
    '''
    start paddle pserver and trainer
    '''
    program = 'paddle train'
    args = " --nics=" + PADDLE_NIC
    args += " --port=" + str(PADDLE_PORT)
    args += " --ports_num=" + str(PADDLE_PORTS_NUM)
    args += " --comment=" + "paddle_process_by_paddle"
    ip_string = ""
    for ip in idMap.keys():
        ip_string += (ip + ",")
    ip_string = ip_string.rstrip(",")
    args += " --pservers=" + ip_string
    args_ext = ""
    for key, value in train_args_dict.items():
        args_ext += (' --' + key + '=' + value)
    localIP = socket.gethostbyname(socket.gethostname())
    trainerId = idMap[localIP]
    args += " " + args_ext + " --trainer_id=" + \
        str(trainerId) + " --save_dir=" + JOB_PATH_OUTPUT
    logDir = JOB_PATH_OUTPUT + "/node_" + str(trainerId)
    if not os.path.exists(JOB_PATH_OUTPUT):
        os.makedirs(JOB_PATH_OUTPUT)
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    copyCommand = 'cp -rf ' + JOB_PATH + \
        "/" + str(trainerId) + "/data/*" + " ./data/"
    os.system(copyCommand)
    startPserver = 'nohup paddle pserver' + \
        " --port=" + str(PADDLE_PORT) + \
        " --ports_num=" + str(PADDLE_PORTS_NUM) + \
        " --ports_num_for_sparse=" + str(PADDLE_PORTS_NUM_SPARSE) + \
        " --nics=" + PADDLE_NIC + \
        " --comment=" + "paddle_process_by_paddle" + \
        " --num_gradient_servers=" + str(PADDLE_SERVER_NUM) +\
        " > " + logDir + "/server.log 2>&1 &"
    print startPserver
    os.system(startPserver)
    # wait until pservers completely start
    time.sleep(20)
    startTrainer = program + args + " 2>&1 | tee " + \
        logDir + "/train.log"
    print startTrainer
    os.system(startTrainer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="start_paddle.py", description='simple tool for k8s')
    args, train_args_list = parser.parse_known_args()
    train_args = refine_unknown_args(train_args_list)
    train_args_dict = dict(zip(train_args[:-1:2], train_args[1::2]))
    podlist = getPodList()
    # need to wait until all pods are running
    while not isPodAllRunning(podlist):
        time.sleep(20)
        podlist = getPodList()
    idMap = getIdMap(podlist)
    startPaddle(idMap, train_args_dict)
