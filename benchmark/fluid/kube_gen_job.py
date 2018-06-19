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

import yaml
import copy
import argparse
import random
import os
import copy
from kube_templates import pserver, trainer, envs


def parse_args():
    parser = argparse.ArgumentParser(description='Generate dist job yamls.')

    parser.add_argument(
        '--jobname', default="paddlejob", help='unique job name')
    parser.add_argument(
        '--cpu', default=1, type=int, help='CPU cores per trainer node')
    parser.add_argument(
        '--pscpu', default=1, type=int, help='CPU cores per pserver node')
    parser.add_argument(
        '--gpu', default=0, type=int, help='num of GPUs per node')
    parser.add_argument(
        '--image',
        default="bootstrapper:5000/fluid_benchmark:gpu",
        help='num of GPUs per node')
    parser.add_argument(
        '--pservers', default=1, type=int, help='num of pservers')
    parser.add_argument(
        '--trainers', default=1, type=int, help='num of trainers')
    parser.add_argument('--memory', default=1, type=int, help='trainer memory')
    parser.add_argument(
        '--psmemory', default=1, type=int, help='pserver memory')
    parser.add_argument(
        '--port', default=30236, type=int, help='num of trainers')
    parser.add_argument(
        '--entry', default="python train.py", help='command to run')
    parser.add_argument(
        '--fluid', default=1, type=int, help='whether is fluid job')
    parser.add_argument(
        '--rdma', action='store_true', help='whether mount rdma libs')
    parser.add_argument(
        '--disttype',
        default="pserver",
        type=str,
        choices=['pserver', 'nccl2', 'local'],
        help='pserver or nccl2 or local')

    args = parser.parse_args()
    return args


def gen_job():
    ps = pserver
    tn = trainer
    args = parse_args()

    ps_container = ps["spec"]["template"]["spec"]["containers"][0]
    tn_container = tn["spec"]["template"]["spec"]["containers"][0]

    if args.fluid == 1:
        ps_container["command"] = \
            ["paddle_k8s", "start_fluid"]
        tn_container["command"] = \
            ["paddle_k8s", "start_fluid"]
    ps["metadata"]["name"] = args.jobname + "-pserver"
    ps["spec"]["template"]["metadata"]["labels"][
        "paddle-job-pserver"] = args.jobname
    tn["metadata"]["name"] = args.jobname + "-trainer"
    tn["spec"]["template"]["metadata"]["labels"]["paddle-job"] = args.jobname

    ps_container["image"] = args.image
    tn_container["image"] = args.image

    ps_container["resources"]["requests"]["cpu"] = str(args.pscpu)
    ps_container["resources"]["requests"]["memory"] = str(args.psmemory) + "Gi"
    ps_container["resources"]["limits"]["cpu"] = str(args.pscpu)
    ps_container["resources"]["limits"]["memory"] = str(args.psmemory) + "Gi"

    tn_container["resources"]["requests"]["cpu"] = str(args.cpu)
    tn_container["resources"]["requests"]["memory"] = str(args.memory) + "Gi"
    tn_container["resources"]["limits"]["cpu"] = str(args.cpu)
    tn_container["resources"]["limits"]["memory"] = str(args.memory) + "Gi"
    if args.gpu > 0:
        tn_container["resources"]["requests"][
            "alpha.kubernetes.io/nvidia-gpu"] = str(args.gpu)
        tn_container["resources"]["limits"][
            "alpha.kubernetes.io/nvidia-gpu"] = str(args.gpu)

    ps["spec"]["replicas"] = int(args.pservers)
    tn["spec"]["parallelism"] = int(args.trainers)
    tn["spec"]["completions"] = int(args.trainers)
    ps_container["ports"][0]["name"] = "jobport-" + str(args.port)
    ps_container["ports"][0]["containerPort"] = args.port
    spreadport = random.randint(40000, 60000)
    tn_container["ports"][0]["name"] = "spr-" + str(spreadport)
    tn_container["ports"][0]["containerPort"] = spreadport

    envs.append({"name": "PADDLE_JOB_NAME", "value": args.jobname})
    envs.append({"name": "PADDLE_TRAINERS", "value": str(args.trainers)})
    envs.append({"name": "PADDLE_PSERVERS", "value": str(args.pservers)})
    envs.append({"name": "ENTRY", "value": args.entry})
    envs.append({"name": "PADDLE_PSERVER_PORT", "value": str(args.port)})
    # NOTE: these directories below are cluster specific, please modify
    # this settings before you run on your own cluster.
    envs.append({
        "name": "LD_LIBRARY_PATH",
        "value":
        "/usr/local/lib:/usr/local/nvidia/lib64:/usr/local/rdma/lib64:/usr/lib64/mlnx_ofed/valgrind"
    })

    volumes = [{
        "name": "nvidia-driver",
        "hostPath": {
            "path": "/usr/local/nvidia/lib64"
        }
    }]
    volumeMounts = [{
        "mountPath": "/usr/local/nvidia/lib64",
        "name": "nvidia-driver"
    }]

    if args.rdma:
        volumes.extend([{
            "name": "ibetc",
            "hostPath": {
                "path": "/etc/libibverbs.d"
            }
        }, {
            "name": "iblibs",
            "hostPath": {
                "path": "/usr/local/rdma"
            }
        }, {
            "name": "valgrind",
            "hostPath": {
                "path": "/usr/lib64/mlnx_ofed/valgrind"
            }
        }])
        volumeMounts.extend([{
            "mountPath": "/etc/libibverbs.d",
            "name": "ibetc"
        }, {
            "mountPath": "/usr/local/rdma",
            "name": "iblibs"
        }, {
            "mountPath": "/usr/lib64/mlnx_ofed/valgrind",
            "name": "valgrind"
        }])
        # append shm for NCCL2
        volumes.append({"name": "dshm", "emptyDir": {"medium": "Memory"}})
        volumeMounts.append({"mountPath": "/dev/shm", "name": "dshm"})

    tn["spec"]["template"]["spec"]["volumes"] = volumes
    tn_container["volumeMounts"] = volumeMounts

    ps_container["env"] = copy.deepcopy(envs)
    ps_container["env"].append({
        "name": "PADDLE_TRAINING_ROLE",
        "value": "PSERVER"
    })
    tn_container["env"] = envs
    if args.disttype == "pserver":
        tn_container["env"].append({
            "name": "PADDLE_TRAINING_ROLE",
            "value": "TRAINER"
        })
    elif args.disttype == "nccl2" or args.disttype == "local":
        # NCCL2 have no training role, set to plain WORKER
        tn_container["env"].append({
            "name": "PADDLE_TRAINING_ROLE",
            "value": "WORKER"
        })

    os.mkdir(args.jobname)
    if args.disttype == "pserver":
        with open("%s/pserver.yaml" % args.jobname, "w") as fn:
            yaml.dump(ps, fn)

    with open("%s/trainer.yaml" % args.jobname, "w") as fn:
        yaml.dump(tn, fn)


if __name__ == "__main__":
    gen_job()
