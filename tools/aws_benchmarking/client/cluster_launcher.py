#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import time
import math
import logging
import copy

import netaddr
import boto3
import namesgenerator
import paramiko
from scp import SCPClient
import requests


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--key_name', type=str, default="", help="required, key pair name")
parser.add_argument(
    '--security_group_id',
    type=str,
    default="",
    help="required, the security group id associated with your VPC")

parser.add_argument(
    '--vpc_id',
    type=str,
    default="",
    help="The VPC in which you wish to run test")
parser.add_argument(
    '--subnet_id',
    type=str,
    default="",
    help="The Subnet_id in which you wish to run test")

parser.add_argument(
    '--pserver_instance_type',
    type=str,
    default="c5.2xlarge",
    help="your pserver instance type, c5.2xlarge by default")
parser.add_argument(
    '--trainer_instance_type',
    type=str,
    default="p2.8xlarge",
    help="your trainer instance type, p2.8xlarge by default")

parser.add_argument(
    '--task_name',
    type=str,
    default="",
    help="the name you want to identify your job")
parser.add_argument(
    '--pserver_image_id',
    type=str,
    default="ami-da2c1cbf",
    help="ami id for system image, default one has nvidia-docker ready, \
    use ami-1ae93962 for us-east-2")

parser.add_argument(
    '--pserver_command',
    type=str,
    default="",
    help="pserver start command, format example: python,vgg.py,batch_size:128,is_local:yes"
)

parser.add_argument(
    '--trainer_image_id',
    type=str,
    default="ami-da2c1cbf",
    help="ami id for system image, default one has nvidia-docker ready, \
    use ami-1ae93962 for us-west-2")

parser.add_argument(
    '--trainer_command',
    type=str,
    default="",
    help="trainer start command, format example: python,vgg.py,batch_size:128,is_local:yes"
)

parser.add_argument(
    '--availability_zone',
    type=str,
    default="us-east-2a",
    help="aws zone id to place ec2 instances")

parser.add_argument(
    '--trainer_count', type=int, default=1, help="Trainer count")

parser.add_argument(
    '--pserver_count', type=int, default=1, help="Pserver count")

parser.add_argument(
    '--action', type=str, default="create", help="create|cleanup|status")

parser.add_argument('--pem_path', type=str, help="private key file")

parser.add_argument(
    '--pserver_port', type=str, default="5436", help="pserver port")

parser.add_argument(
    '--docker_image', type=str, default="busybox", help="training docker image")

parser.add_argument(
    '--master_server_port', type=int, default=5436, help="master server port")

parser.add_argument(
    '--master_server_public_ip', type=str, help="master server public ip")

parser.add_argument(
    '--master_docker_image',
    type=str,
    default="putcn/paddle_aws_master:latest",
    help="master docker image id")

parser.add_argument(
    '--no_clean_up',
    type=str2bool,
    default=False,
    help="whether to clean up after training")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

ec2client = boto3.client('ec2')


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def create_subnet():
    # if no vpc id provided, list vpcs
    logging.info("start creating subnet")
    if not args.vpc_id:
        logging.info("no vpc provided, trying to find the default one")
        vpcs_desc = ec2client.describe_vpcs(
            Filters=[{
                "Name": "isDefault",
                "Values": ["true", ]
            }], )
        if len(vpcs_desc["Vpcs"]) == 0:
            raise ValueError('No default VPC')
        args.vpc_id = vpcs_desc["Vpcs"][0]["VpcId"]
        vpc_cidrBlock = vpcs_desc["Vpcs"][0]["CidrBlock"]

        logging.info("default vpc fount with id %s and CidrBlock %s" %
                     (args.vpc_id, vpc_cidrBlock))

    if not vpc_cidrBlock:
        logging.info("trying to find cidrblock for vpc")
        vpcs_desc = ec2client.describe_vpcs(
            Filters=[{
                "Name": "vpc-id",
                "Values": [args.vpc_id, ],
            }], )
        if len(vpcs_desc["Vpcs"]) == 0:
            raise ValueError('No VPC found')
        vpc_cidrBlock = vpcs_desc["Vpcs"][0]["CidrBlock"]
        logging.info("cidrblock for vpc is %s" % vpc_cidrBlock)

    # list subnets in vpc in order to create a new one

    logging.info("trying to find ip blocks for new subnet")
    subnets_desc = ec2client.describe_subnets(
        Filters=[{
            "Name": "vpc-id",
            "Values": [args.vpc_id, ],
        }], )

    ips_taken = []
    for subnet_dec in subnets_desc["Subnets"]:
        ips_taken.append(subnet_dec["CidrBlock"])

    ip_blocks_avaliable = netaddr.IPSet(
        [vpc_cidrBlock]) ^ netaddr.IPSet(ips_taken)
    # adding 10 addresses as buffer
    cidr_prefix = 32 - math.ceil(
        math.log(args.pserver_count + args.trainer_count + 10, 2))
    if cidr_prefix <= 16:
        raise ValueError('Too many nodes to fit in current VPC')

    for ipnetwork in ip_blocks_avaliable.iter_cidrs():
        try:
            subnet_cidr = ipnetwork.subnet(int(cidr_prefix)).next()
            logging.info("subnet ip block found %s" % (subnet_cidr))
            break
        except Exception:
            pass

    if not subnet_cidr:
        raise ValueError(
            'No avaliable subnet to fit required nodes in current VPC')

    logging.info("trying to create subnet")
    subnet_desc = ec2client.create_subnet(
        CidrBlock=str(subnet_cidr),
        VpcId=args.vpc_id,
        AvailabilityZone=args.availability_zone)

    subnet_id = subnet_desc["Subnet"]["SubnetId"]

    subnet_waiter = ec2client.get_waiter('subnet_available')
    # sleep for 1s before checking its state
    time.sleep(1)
    subnet_waiter.wait(SubnetIds=[subnet_id, ])

    logging.info("subnet created")

    logging.info("adding tags to newly created subnet")
    ec2client.create_tags(
        Resources=[subnet_id, ],
        Tags=[{
            "Key": "Task_name",
            'Value': args.task_name
        }])
    return subnet_id


def run_instances(image_id, instance_type, count=1, role="MASTER", cmd=""):
    response = ec2client.run_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        MaxCount=count,
        MinCount=count,
        UserData=cmd,
        DryRun=False,
        InstanceInitiatedShutdownBehavior="stop",
        KeyName=args.key_name,
        Placement={'AvailabilityZone': args.availability_zone},
        NetworkInterfaces=[{
            'DeviceIndex': 0,
            'SubnetId': args.subnet_id,
            "AssociatePublicIpAddress": True,
            'Groups': args.security_group_ids
        }],
        TagSpecifications=[{
            'ResourceType': "instance",
            'Tags': [{
                "Key": 'Task_name',
                "Value": args.task_name + "_master"
            }, {
                "Key": 'Role',
                "Value": role
            }]
        }])

    instance_ids = []
    for instance in response["Instances"]:
        instance_ids.append(instance["InstanceId"])

    if len(instance_ids) > 0:
        logging.info(str(len(instance_ids)) + " instance(s) created")
    else:
        logging.info("no instance created")
    #create waiter to make sure it's running

    logging.info("waiting for instance to become accessible")
    waiter = ec2client.get_waiter('instance_status_ok')
    waiter.wait(
        Filters=[{
            "Name": "instance-status.status",
            "Values": ["ok"]
        }, {
            "Name": "instance-status.reachability",
            "Values": ["passed"]
        }, {
            "Name": "instance-state-name",
            "Values": ["running"]
        }],
        InstanceIds=instance_ids)

    instances_response = ec2client.describe_instances(InstanceIds=instance_ids)

    return instances_response["Reservations"][0]["Instances"]


def generate_task_name():
    return namesgenerator.get_random_name()


def init_args():

    if not args.task_name:
        args.task_name = generate_task_name()
        logging.info("task name generated %s" % (args.task_name))

    if not args.pem_path:
        args.pem_path = os.path.expanduser("~") + "/" + args.key_name + ".pem"
    if args.security_group_id:
        args.security_group_ids = (args.security_group_id, )


def create():

    init_args()

    # create subnet
    if not args.subnet_id:
        args.subnet_id = create_subnet()

    # create master node

    master_instance_response = run_instances(
        image_id="ami-7a05351f", instance_type="t2.nano")

    logging.info("master server started")

    args.master_server_public_ip = master_instance_response[0][
        "PublicIpAddress"]
    args.master_server_ip = master_instance_response[0]["PrivateIpAddress"]

    logging.info("master server started, master_ip=%s, task_name=%s" %
                 (args.master_server_public_ip, args.task_name))

    # cp config file and pems to master node

    ssh_key = paramiko.RSAKey.from_private_key_file(args.pem_path)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(
        hostname=args.master_server_public_ip, username="ubuntu", pkey=ssh_key)

    with SCPClient(ssh_client.get_transport()) as scp:
        scp.put(os.path.expanduser("~") + "/" + ".aws",
                recursive=True,
                remote_path='/home/ubuntu/')
        scp.put(args.pem_path,
                remote_path='/home/ubuntu/' + args.key_name + ".pem")

    logging.info("credentials and pem copied to master")

    # set arguments and start docker
    kick_off_cmd = "docker run -d -v /home/ubuntu/.aws:/root/.aws/"
    kick_off_cmd += " -v /home/ubuntu/" + args.key_name + ".pem:/root/" + args.key_name + ".pem"
    kick_off_cmd += " -v /home/ubuntu/logs/:/root/logs/"
    kick_off_cmd += " -p " + str(args.master_server_port) + ":" + str(
        args.master_server_port)
    kick_off_cmd += " " + args.master_docker_image

    args_to_pass = copy.copy(args)
    args_to_pass.action = "serve"
    del args_to_pass.pem_path
    del args_to_pass.security_group_ids
    del args_to_pass.master_docker_image
    del args_to_pass.master_server_public_ip
    for arg, value in sorted(vars(args_to_pass).iteritems()):
        if value:
            kick_off_cmd += ' --%s %s' % (arg, value)

    logging.info(kick_off_cmd)
    stdin, stdout, stderr = ssh_client.exec_command(command=kick_off_cmd)
    return_code = stdout.channel.recv_exit_status()
    logging.info(return_code)
    if return_code != 0:
        raise Exception("Error while kicking off master")

    logging.info(
        "master server finished init process, visit %s to check master log" %
        (get_master_web_url("/status")))


def cleanup():
    print requests.post(get_master_web_url("/cleanup")).text


def status():
    print requests.post(get_master_web_url("/status")).text


def get_master_web_url(path):
    return "http://" + args.master_server_public_ip + ":" + str(
        args.master_server_port) + path


if __name__ == "__main__":
    print_arguments()
    if args.action == "create":
        if not args.key_name or not args.security_group_id:
            raise ValueError("key_name and security_group_id are required")
        create()
    elif args.action == "cleanup":
        if not args.master_server_public_ip:
            raise ValueError("master_server_public_ip is required")
        cleanup()
    elif args.action == "status":
        if not args.master_server_public_ip:
            raise ValueError("master_server_public_ip is required")
        status()
