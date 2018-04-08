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
import threading
import logging

import netaddr
import boto3
import namesgenerator
import paramiko

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

# You must have aws_access_key_id, aws_secret_access_key, region set in
# ~/.aws/credentials and ~/.aws/config

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
    default="p2.8xlarge",
    help="your pserver instance type, p2.8xlarge by default")
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
    help="ami id for system image, default one has nvidia-docker ready, use ami-1ae93962 for us-east-2"
)
parser.add_argument(
    '--trainer_image_id',
    type=str,
    default="ami-da2c1cbf",
    help="ami id for system image, default one has nvidia-docker ready, use ami-1ae93962 for us-west-2"
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
    '--pserver_bash_file',
    type=str,
    default=os.path.join(os.path.dirname(__file__), "pserver.sh.template"),
    help="pserver bash file path")

parser.add_argument(
    '--trainer_bash_file',
    type=str,
    default=os.path.join(os.path.dirname(__file__), "trainer.sh.template"),
    help="trainer bash file path")

parser.add_argument(
    '--action', type=str, default="serve", help="create|cleanup|serve")

parser.add_argument('--pem_path', type=str, help="private key file")

parser.add_argument(
    '--pserver_port', type=str, default="5436", help="pserver port")

parser.add_argument(
    '--docker_image', type=str, default="busybox", help="training docker image")

parser.add_argument(
    '--master_server_port', type=int, default=5436, help="master server port")

parser.add_argument(
    '--master_server_ip', type=str, default="", help="master server private ip")

args = parser.parse_args()

ec2client = boto3.client('ec2')

logging.basicConfig(
    filename='master.log', level=logging.INFO, format='%(asctime)s %(message)s')


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


def create_pservers():
    try:
        return run_instances(
            image_id=args.pserver_image_id,
            instance_type=args.pserver_instance_type,
            count=args.pserver_count,
            role="PSERVER", )
    except Exception:
        logging.exception("error while trying to create pservers")
        cleanup(args.task_name)


def create_trainers(kickoff_cmd, pserver_endpoints_str):
    try:
        responses = []
        for i in xrange(args.trainer_count):
            cmd = kickoff_cmd.format(
                PSERVER_HOSTS=pserver_endpoints_str,
                DOCKER_IMAGE=args.docker_image,
                TRAINER_INDEX=str(i),
                TASK_NAME=args.task_name,
                MASTER_ENDPOINT=args.master_server_ip + ":" +
                str(args.master_server_port))
            logging.info(cmd)
            responses.append(
                run_instances(
                    image_id=args.trainer_image_id,
                    instance_type=args.trainer_instance_type,
                    count=1,
                    role="TRAINER",
                    cmd=cmd, )[0])
        return responses
    except Exception:
        logging.exception("error while trying to create trainers")
        cleanup(args.task_name)


def cleanup(task_name):
    #shutdown all ec2 instances
    print("going to clean up " + task_name + " instances")
    instances_response = ec2client.describe_instances(Filters=[{
        "Name": "tag:Task_name",
        "Values": [task_name]
    }])

    instance_ids = []
    if len(instances_response["Reservations"]) > 0:
        for reservation in instances_response["Reservations"]:
            for instance in reservation["Instances"]:
                instance_ids.append(instance["InstanceId"])

        ec2client.terminate_instances(InstanceIds=instance_ids)

        instance_termination_waiter = ec2client.get_waiter(
            'instance_terminated')
        instance_termination_waiter.wait(InstanceIds=instance_ids)

    #delete the subnet created

    subnet = ec2client.describe_subnets(Filters=[{
        "Name": "tag:Task_name",
        "Values": [task_name]
    }])

    if len(subnet["Subnets"]) > 0:
        ec2client.delete_subnet(SubnetId=subnet["Subnets"][0]["SubnetId"])
    # no subnet delete waiter, just leave it.
    logging.info("Clearnup done")
    return


def kickoff_pserver(host, pserver_endpoints_str):
    try:
        ssh_key = paramiko.RSAKey.from_private_key_file(args.pem_path)
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=host, username="ubuntu", pkey=ssh_key)
        cmd = (script_to_str(args.pserver_bash_file)).format(
            PSERVER_HOSTS=pserver_endpoints_str,
            DOCKER_IMAGE=args.docker_image,
            PSERVER_PORT=args.pserver_port,
            TASK_NAME=args.task_name,
            MASTER_ENDPOINT=args.master_server_ip + ":" +
            str(args.master_server_port))
        logging.info(cmd)
        stdin, stdout, stderr = ssh_client.exec_command(command=cmd)
        return_code = stdout.channel.recv_exit_status()
        logging.info(return_code)
        if return_code != 0:
            raise Exception("Error while kicking off pserver training process")
    except Exception:
        logging.exception("Error while kicking off pserver training process")
        cleanup(args.task_name)
    finally:
        ssh_client.close()


def init_args():

    if not args.task_name:
        args.task_name = generate_task_name()
        logging.info("task name generated %s" % (args.task_name))

    if not args.pem_path:
        args.pem_path = os.path.expanduser("~") + "/" + args.key_name + ".pem"
    if args.security_group_id:
        args.security_group_ids = (args.security_group_id, )

    args.trainers_job_done_count = 0


def create_cluster():

    if not args.subnet_id:
        logging.info("creating subnet for this task")
        args.subnet_id = create_subnet()
        logging.info("subnet %s created" % (args.subnet_id))

    logging.info("creating pservers")
    pserver_create_response = create_pservers()
    logging.info("pserver created, collecting pserver ips")

    pserver_endpoints = []
    for pserver in pserver_create_response:
        pserver_endpoints.append(pserver["NetworkInterfaces"][0][
            "PrivateIpAddress"] + ":" + args.pserver_port)

    pserver_endpoints_str = ",".join(pserver_endpoints)

    logging.info("kicking off pserver training process")
    pserver_threads = []
    for pserver in pserver_create_response:
        pserver_thread = threading.Thread(
            target=kickoff_pserver,
            args=(pserver["PublicIpAddress"], pserver_endpoints_str))
        pserver_thread.start()
        pserver_threads.append(pserver_thread)

    for pserver_thread in pserver_threads:
        pserver_thread.join()

    logging.info("all pserver training process started")

    logging.info("creating trainers and kicking off trainer training process")
    create_trainers(
        kickoff_cmd=script_to_str(args.trainer_bash_file),
        pserver_endpoints_str=pserver_endpoints_str)
    logging.info("trainers created")


def start_server(args):
    class S(BaseHTTPRequestHandler):
        def _set_headers(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/text')
            self.end_headers()

        def do_HEAD(self):
            self._set_headers()

        def do_404(self):
            self.send_response(404)
            self.send_header('Content-type', 'text/text')
            self.end_headers()
            logging.info("Received invalid GET request" + self.path)
            self.wfile.write("NO ACTION FOUND")

        def do_GET(self):
            self._set_headers()
            request_path = self.path
            if request_path == "/status" or request_path == "/logs":
                logging.info("Received request to return status")
                with open("master.log", "r") as logfile:
                    self.wfile.write(logfile.read().strip())
            else:
                self.do_404()

        def do_POST(self):

            request_path = self.path

            if request_path == "/save_data":
                self._set_headers()
                logging.info("Received request to save data")
                self.wfile.write("DATA SAVED!")
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                if args.task_name:
                    with open(args.task_name + ".txt", "a") as text_file:
                        text_file.write(post_data + "\n")

            elif request_path == "/cleanup":
                self._set_headers()
                logging.info("Received request to cleanup cluster")
                cleanup(args.task_name)
                self.wfile.write("cleanup in progress")

            elif request_path == "/trainer_job_done":
                self._set_headers()
                logging.info("Received request to increase job done count")
                args.trainers_job_done_count += 1
                self.wfile.write(
                    str(args.trainers_job_done_count) + " tainers job done")
                if args.trainers_job_done_count >= args.trainer_count:
                    logging.info("going to clean up")
                    cleanup(args.task_name)

            else:
                self.do_404()

    server_address = ('', args.master_server_port)
    httpd = HTTPServer(server_address, S)
    logging.info("HTTP server is starting")
    httpd.serve_forever()


def print_arguments():
    logging.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        logging.info('%s: %s' % (arg, value))
    logging.info('------------------------------------------------')


if __name__ == "__main__":
    print_arguments()
    if args.action == "create":
        logging.info("going to create cluster")
        if not args.key_name or not args.security_group_id:
            raise ValueError("key_name and security_group_id are required")
        init_args()
        create_cluster()
    elif args.action == "cleanup":
        logging.info("going to cleanup cluster")
        if not args.task_name:
            raise ValueError("task_name is required")
        cleanup(args.task_name)
    elif args.action == "serve":
        # serve mode
        if not args.master_server_ip:
            raise ValueError(
                "No master server ip set, please run with --action create")

        logging.info("going to start serve and create cluster")

        init_args()

        logging.info("starting server in another thread")
        server_thread = threading.Thread(target=start_server, args=(args, ))
        server_thread.start()

        create_cluster()
        server_thread.join()
