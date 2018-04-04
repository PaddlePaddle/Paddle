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
import json
import math
import time
import base64

import netaddr
import boto3
import namesgenerator
import paramiko

# You must have aws_access_key_id, aws_secret_access_key, region set in
# ~/.aws/credentials and ~/.aws/config

parser = argparse.ArgumentParser(description=__doc__)
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
    '--security_group_id',
    type=str,
    default="",
    required=True,
    help="required, the security group id associated with your VPC")
parser.add_argument(
    '--pserver_instance_type',
    type=str,
    default="p2.xlarge",
    help="your pserver instance type")
parser.add_argument(
    '--trainer_instance_type',
    type=str,
    default="p2.xlarge",
    help="your trainer instance type")
parser.add_argument(
    '--key_name',
    type=str,
    default="",
    required=True,
    help="required, key pair name")
parser.add_argument(
    '--task_name',
    type=str,
    default="",
    help="the name you want to identify your job")
parser.add_argument(
    '--pserver_image_id',
    type=str,
    default="ami-1ae93962",
    help="ami id for system image, default one has nvidia-docker ready")
parser.add_argument(
    '--trainer_image_id',
    type=str,
    default="ami-1ae93962",
    help="ami id for system image, default one has nvidia-docker ready")

parser.add_argument(
    '--trainer_count', type=int, default=1, help="Trainer count")

parser.add_argument(
    '--pserver_count', type=int, default=1, help="Pserver count")

parser.add_argument(
    '--pserver_bash_file',
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "pserver.sh.template"),
    help="pserver bash file path")

parser.add_argument(
    '--trainer_bash_file',
    type=str,
    required=False,
    default=os.path.join(os.path.dirname(__file__), "trainer.sh.template"),
    help="trainer bash file path")

parser.add_argument('--pem_path', type=str, help="private key file")

parser.add_argument(
    '--pserver_port', type=str, default="5436", help="pserver port")

parser.add_argument(
    '--docker_image', type=str, default="busybox", help="training docker image")

args = parser.parse_args()

ec2client = boto3.client('ec2')


def create_subnet():
    # if no vpc id provided, list vpcs
    if not args.vpc_id:
        print("no vpc provided, trying to find the default one")
        vpcs_desc = ec2client.describe_vpcs(
            Filters=[{
                "Name": "isDefault",
                "Values": ["true", ]
            }], )
        if len(vpcs_desc["Vpcs"]) == 0:
            raise ValueError('No default VPC')
        args.vpc_id = vpcs_desc["Vpcs"][0]["VpcId"]
        vpc_cidrBlock = vpcs_desc["Vpcs"][0]["CidrBlock"]

        print("default vpc fount with id %s and CidrBlock %s" %
              (args.vpc_id, vpc_cidrBlock))

    if not vpc_cidrBlock:
        print("trying to find cidrblock for vpc")
        vpcs_desc = ec2client.describe_vpcs(
            Filters=[{
                "Name": "vpc-id",
                "Values": [args.vpc_id, ],
            }], )
        if len(vpcs_desc["Vpcs"]) == 0:
            raise ValueError('No VPC found')
        vpc_cidrBlock = vpcs_desc["Vpcs"][0]["CidrBlock"]
        print("cidrblock for vpc is %s" % vpc_cidrBlock)

    # list subnets in vpc in order to create a new one

    print("trying to find ip blocks for new subnet")
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
            print("subnet ip block found %s" % (subnet_cidr))
            break
        except Exception:
            pass

    if not subnet_cidr:
        raise ValueError(
            'No avaliable subnet to fit required nodes in current VPC')

    print("trying to create subnet")
    subnet_desc = ec2client.create_subnet(
        CidrBlock=str(subnet_cidr), VpcId=args.vpc_id)

    subnet_id = subnet_desc["Subnet"]["SubnetId"]

    subnet_waiter = ec2client.get_waiter('subnet_available')
    # sleep for 1s before checking its state
    time.sleep(1)
    subnet_waiter.wait(SubnetIds=[subnet_id, ])

    print("subnet created")

    print("adding tags to newly created subnet")
    ec2client.create_tags(
        Resources=[subnet_id, ],
        Tags=[{
            "Key": "Task_name",
            'Value': args.task_name
        }])
    return subnet_id


def generate_task_name():
    return namesgenerator.get_random_name()


def script_to_str(file_path):
    if not file_path:
        return "echo $PSERVER_HOSTS"
    file = open(file_path, 'r')
    text = file.read().strip()
    file.close()
    return text


def run_instances(image_id, instance_type, count, role, cmd=""):
    if cmd:
        cmd = base64.b64encode(cmd)
    response = ec2client.run_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        MaxCount=count,
        MinCount=count,
        UserData=cmd,
        DryRun=False,
        InstanceInitiatedShutdownBehavior="stop",
        KeyName=args.key_name,
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
                "Value": args.task_name
            }, {
                "Key": 'Role',
                "Value": role
            }]
        }])

    instance_ids = []
    for instance in response["Instances"]:
        instance_ids.append(instance["InstanceId"])

    if len(instance_ids) > 0:
        print(str(len(instance_ids)) + " instance(s) created")
    else:
        print("no instance created")
    #create waiter to make sure it's running

    print("waiting for instance to become accessible")
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


def create_pservers():
    return run_instances(
        image_id=args.pserver_image_id,
        instance_type=args.pserver_instance_type,
        count=args.pserver_count,
        role="PSERVER", )


def create_trainers(kickoff_cmd, pserver_endpoints_str):
    responses = []
    for i in xrange(args.trainer_count):
        cmd = kickoff_cmd.format(
            PSERVER_HOSTS=pserver_endpoints_str,
            DOCKER_IMAGE=args.docker_image,
            TRAINER_INDEX=str(i))
        print(cmd)
        responses.append(
            run_instances(
                image_id=args.trainer_image_id,
                instance_type=args.trainer_instance_type,
                count=1,
                role="TRAINER",
                cmd=cmd, )[0])
    return responses


def cleanup(task_name):
    #shutdown all ec2 instances
    instances = ec2client.describe_instances(Filters=[{
        "Name": "tag",
        "Value": "Task_name=" + task_name
    }])

    instance_ids = []
    for instance in instances["Reservations"][0]["Instances"]:
        instance_ids.append(instance["InstanceId"])

    ec2client.stop_instances(InstanceIds=instance_ids)

    instance_stop_waiter = ec2client.get_waiter('instance_stopped')
    instance_stop_waiter.wait(InstanceIds=instance_ids)

    #delete the subnet created

    subnet = ec2client.describe_subnets(Filters=[{
        "Name": "tag",
        "Value": "Task_name=" + task_name
    }])

    ec2client.delete_subnet(SubnetId=subnet["Subnets"][0]["SubnetId"])

    # no subnet delete waiter, just leave it.

    return


def main():
    if not args.task_name:
        args.task_name = generate_task_name()
        print("task name generated", args.task_name)

    if not args.subnet_id:
        print("creating subnet for this task")
        args.subnet_id = create_subnet()
        print("subnet %s created" % (args.subnet_id))

    if not args.pem_path:
        args.pem_path = os.path.expanduser("~") + "/" + args.key_name + ".pem"
    if args.security_group_id:
        args.security_group_ids = (args.security_group_id, )

    print("creating pservers")
    pserver_create_response = create_pservers()
    print("pserver created, collecting pserver ips")

    pserver_endpoints = []
    for pserver in pserver_create_response:
        pserver_endpoints.append(pserver["NetworkInterfaces"][0][
            "PrivateIpAddress"] + ":" + args.pserver_port)

    pserver_endpoints_str = ",".join(pserver_endpoints)

    # ssh to pservers to start training
    ssh_key = paramiko.RSAKey.from_private_key_file(args.pem_path)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("kicking off pserver training process")
    for pserver in pserver_create_response:
        try:
            ssh_client.connect(
                hostname=pserver["PublicIpAddress"],
                username="ubuntu",
                pkey=ssh_key)
            cmd = (script_to_str(args.pserver_bash_file)).format(
                PSERVER_HOSTS=pserver_endpoints_str,
                DOCKER_IMAGE=args.docker_image)
            print(cmd)
            stdin, stdout, stderr = ssh_client.exec_command(command=cmd)
            if stderr.read():
                raise Exception(
                    "Error while kicking off pserver training process")
            #print(stdout.read())
        except Exception, e:
            print e
            cleanup(args.task_name)
        finally:
            ssh_client.close()

    print("creating trainers and kicking off trainer training process")
    create_trainers(
        kickoff_cmd=script_to_str(args.trainer_bash_file),
        pserver_endpoints_str=pserver_endpoints_str)


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    print_arguments()
    main()
