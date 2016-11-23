#PaddlePaddle on AWS with Kubernetes

##Prerequisites

You need an Amazon account and your user account needs the following privileges to continue:

* AmazonEC2FullAccess
* AmazonS3FullAccess
* AmazonRoute53FullAccess
* AmazonRoute53DomainsFullAccess
* AmazonVPCFullAccess
* IAMUserSSHKeys
* IAMFullAccess
* NetworkAdministrator

If you are not in Unites States, we also recommend creating a jump server instance with default amazon AMI in the same available zone as your cluster, otherwise there will be some issue on creating the cluster.


##For people new to Kubernetes and AWS

If you are new to Kubernetes or AWS and just want to run PaddlePaddle, you can follow these steps to start up a new cluster.

###AWS Login

First configure your aws account information:

```
aws configure

```
Fill in the required fields:

```
AWS Access Key ID: YOUR_ACCESS_KEY_ID
AWS Secrete Access Key: YOUR_SECRETE_ACCESS_KEY
Default region name: us-west-2
Default output format: json

```

###Cluster Start Up
And then type the following command:

```
export KUBERNETES_PROVIDER=aws; curl -sS https://get.k8s.io | bash

```


This process takes about 5 to 10 minutes. 

Once the cluster is up, the IP addresses of your master and node(s) will be printed, as well as information about the default services running in the cluster (monitoring, logging, dns). 

User credentials and security tokens are written in `~/.kube/config`, they will be necessary to use the CLI or the HTTP Basic Auth.


```
[ec2-user@ip-172-31-24-50 ~]$ export KUBERNETES_PROVIDER=aws; curl -sS https://get.k8s.io | bash
'kubernetes' directory already exist. Should we skip download step and start to create cluster based on it? [Y]/n
Skipping download step.
Creating a kubernetes on aws...
... Starting cluster in us-west-2a using provider aws
... calling verify-prereqs
... calling kube-up
Starting cluster using os distro: jessie
Uploading to Amazon S3
+++ Staging server tars to S3 Storage: kubernetes-staging-98b0b8ae5c8ea0e33a0faa67722948f1/devel
upload: ../../../tmp/kubernetes.7nMCAR/s3/bootstrap-script to s3://kubernetes-staging-98b0b8ae5c8ea0e33a0faa67722948f1/devel/bootstrap-script
Uploaded server tars:
  SERVER_BINARY_TAR_URL: https://s3.amazonaws.com/kubernetes-staging-98b0b8ae5c8ea0e33a0faa67722948f1/devel/kubernetes-server-linux-amd64.tar.gz
  SALT_TAR_URL: https://s3.amazonaws.com/kubernetes-staging-98b0b8ae5c8ea0e33a0faa67722948f1/devel/kubernetes-salt.tar.gz
  BOOTSTRAP_SCRIPT_URL: https://s3.amazonaws.com/kubernetes-staging-98b0b8ae5c8ea0e33a0faa67722948f1/devel/bootstrap-script
INSTANCEPROFILE	arn:aws:iam::525016323257:instance-profile/kubernetes-master	2016-11-22T05:20:41Z	AIPAJWBAGNSEHM4CILHDY	kubernetes-master	/
ROLES	arn:aws:iam::525016323257:role/kubernetes-master	2016-11-22T05:20:39Z	/	AROAJW3VKVVQ5MZSTTJ5O	kubernetes-master
ASSUMEROLEPOLICYDOCUMENT	2012-10-17
STATEMENT	sts:AssumeRole	Allow
PRINCIPAL	ec2.amazonaws.com
INSTANCEPROFILE	arn:aws:iam::525016323257:instance-profile/kubernetes-minion	2016-11-22T05:20:45Z	AIPAIYVABOPWQZZX5EN5W	kubernetes-minion	/
ROLES	arn:aws:iam::525016323257:role/kubernetes-minion	2016-11-22T05:20:43Z	/	AROAJKDVM7XQNZ4JGVKNO	kubernetes-minion
ASSUMEROLEPOLICYDOCUMENT	2012-10-17
STATEMENT	sts:AssumeRole	Allow
PRINCIPAL	ec2.amazonaws.com
Using SSH key with (AWS) fingerprint: 08:9f:6b:82:3d:b5:ba:a0:f3:db:ab:94:1b:a7:a4:c7
Creating vpc.
Adding tag to vpc-fad1139d: Name=kubernetes-vpc
Adding tag to vpc-fad1139d: KubernetesCluster=kubernetes
Using VPC vpc-fad1139d
Adding tag to dopt-e43a7180: Name=kubernetes-dhcp-option-set
Adding tag to dopt-e43a7180: KubernetesCluster=kubernetes
Using DHCP option set dopt-e43a7180
Creating subnet.
Adding tag to subnet-fc16fa9b: KubernetesCluster=kubernetes
Using subnet subnet-fc16fa9b
Creating Internet Gateway.
Using Internet Gateway igw-fc0d9398
Associating route table.
Creating route table
Adding tag to rtb-bd8512da: KubernetesCluster=kubernetes
Associating route table rtb-bd8512da to subnet subnet-fc16fa9b
Adding route to route table rtb-bd8512da
Using Route Table rtb-bd8512da
Creating master security group.
Creating security group kubernetes-master-kubernetes.
Adding tag to sg-d9280ba0: KubernetesCluster=kubernetes
Creating minion security group.
Creating security group kubernetes-minion-kubernetes.
Adding tag to sg-dc280ba5: KubernetesCluster=kubernetes
Using master security group: kubernetes-master-kubernetes sg-d9280ba0
Using minion security group: kubernetes-minion-kubernetes sg-dc280ba5
Creating master disk: size 20GB, type gp2
Adding tag to vol-04d71a810478dec0d: Name=kubernetes-master-pd
Adding tag to vol-04d71a810478dec0d: KubernetesCluster=kubernetes
Allocated Elastic IP for master: 35.162.175.115
Adding tag to vol-04d71a810478dec0d: kubernetes.io/master-ip=35.162.175.115
Generating certs for alternate-names: IP:35.162.175.115,IP:172.20.0.9,IP:10.0.0.1,DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.cluster.local,DNS:kubernetes-master
Starting Master
Adding tag to i-042488375c2ca1e3e: Name=kubernetes-master
Adding tag to i-042488375c2ca1e3e: Role=kubernetes-master
Adding tag to i-042488375c2ca1e3e: KubernetesCluster=kubernetes
Waiting for master to be ready
Attempt 1 to check for master nodeWaiting for instance i-042488375c2ca1e3e to be running (currently pending)
Sleeping for 3 seconds...
Waiting for instance i-042488375c2ca1e3e to be running (currently pending)
Sleeping for 3 seconds...
Waiting for instance i-042488375c2ca1e3e to be running (currently pending)
Sleeping for 3 seconds...
Waiting for instance i-042488375c2ca1e3e to be running (currently pending)
Sleeping for 3 seconds...
Waiting for instance i-042488375c2ca1e3e to be running (currently pending)
Sleeping for 3 seconds...
 [master running]
Attaching IP 35.162.175.115 to instance i-042488375c2ca1e3e
Attaching persistent data volume (vol-04d71a810478dec0d) to master
2016-11-23T02:14:59.645Z	/dev/sdb	i-042488375c2ca1e3e	attaching	vol-04d71a810478dec0d
cluster "aws_kubernetes" set.
user "aws_kubernetes" set.
context "aws_kubernetes" set.
switched to context "aws_kubernetes".
user "aws_kubernetes-basic-auth" set.
Wrote config for aws_kubernetes to /home/ec2-user/.kube/config
Creating minion configuration
Creating autoscaling group
 0 minions started; waiting
 0 minions started; waiting
 0 minions started; waiting
 0 minions started; waiting
 2 minions started; ready
Waiting for cluster initialization.

  This will continually check to see if the API for kubernetes is reachable.
  This might loop forever if there was some uncaught error during start
  up.

.......................................................................................................................................................................................................................Kubernetes cluster created.
Sanity checking cluster...
Attempt 1 to check Docker on node @ 35.164.79.249 ...working
Attempt 1 to check Docker on node @ 35.164.83.190 ...working

Kubernetes cluster is running.  The master is running at:

  https://35.162.175.115

The user name and password to use is located in /home/ec2-user/.kube/config.

... calling validate-cluster
Waiting for 2 ready nodes. 0 ready nodes, 2 registered. Retrying.
Waiting for 2 ready nodes. 1 ready nodes, 2 registered. Retrying.
Waiting for 2 ready nodes. 1 ready nodes, 2 registered. Retrying.
Found 2 node(s).
NAME                                        STATUS    AGE
ip-172-20-0-23.us-west-2.compute.internal   Ready     54s
ip-172-20-0-24.us-west-2.compute.internal   Ready     52s
Validate output:
NAME                 STATUS    MESSAGE              ERROR
scheduler            Healthy   ok
controller-manager   Healthy   ok
etcd-1               Healthy   {"health": "true"}
etcd-0               Healthy   {"health": "true"}
Cluster validation succeeded
Done, listing cluster services:

Kubernetes master is running at https://35.162.175.115
Elasticsearch is running at https://35.162.175.115/api/v1/proxy/namespaces/kube-system/services/elasticsearch-logging
Heapster is running at https://35.162.175.115/api/v1/proxy/namespaces/kube-system/services/heapster
Kibana is running at https://35.162.175.115/api/v1/proxy/namespaces/kube-system/services/kibana-logging
KubeDNS is running at https://35.162.175.115/api/v1/proxy/namespaces/kube-system/services/kube-dns
kubernetes-dashboard is running at https://35.162.175.115/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard
Grafana is running at https://35.162.175.115/api/v1/proxy/namespaces/kube-system/services/monitoring-grafana
InfluxDB is running at https://35.162.175.115/api/v1/proxy/namespaces/kube-system/services/monitoring-influxdb

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.

Kubernetes binaries at /home/ec2-user/kubernetes/cluster/
You may want to add this directory to your PATH in $HOME/.profile
Installation successful!
```


By default, the script will provision a new VPC and a 4 node k8s cluster in us-west-2a (Oregon) with EC2 instances running on Debian. You can override the variables defined in `<path/to/kubernetes-directory>/cluster/config-default.sh` to change this behavior as follows:

```
export KUBE_AWS_ZONE=us-west-2a 
export NUM_NODES=2 
export MASTER_SIZE=m3.medium 
export NODE_SIZE=m3.medium 
export AWS_S3_REGION=us-west-2a 
export AWS_S3_BUCKET=mycompany-kubernetes-artifacts 
export KUBE_AWS_INSTANCE_PREFIX=k8s 
...

```
And then concate the kubernetes binaries directory into PATH:

```
export PATH=<path/to/kubernetes-directory>/platforms/linux/amd64:$PATH

```
Now you can use administration tool kubectl to operate the cluster.
By default, kubectl will use the kubeconfig file generated during the cluster startup for authenticating against the API, the location is in `~/.kube/config`.

For running PaddlePaddle training with Kubernetes on AWS, you can refer to [this article](https://github.com/drinktee/Paddle/blob/k8s/doc_cn/cluster/k8s/distributed_training_on_kubernetes.md).


###Cluster Tear Down
If you want to tear down the running cluster:

```
export KUBERNETES_PROVIDER=aws; <path/to/kubernetes-directory>/cluster/kube-down.sh
```

This process takes about 2 to 5 minutes.

```
[ec2-user@ip-172-31-24-50 ~]$ export KUBERNETES_PROVIDER=aws; ./kubernetes/cluster/kube-down.sh
Bringing down cluster using provider: aws
Deleting instances in VPC: vpc-fad1139d
Deleting auto-scaling group: kubernetes-minion-group-us-west-2a
Deleting auto-scaling launch configuration: kubernetes-minion-group-us-west-2a
Deleting auto-scaling group: kubernetes-minion-group-us-west-2a
Waiting for instances to be deleted
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
Waiting for instance i-09d7e8824ef1f8384 to be terminated (currently shutting-down)
Sleeping for 3 seconds...
All instances deleted
Releasing Elastic IP: 35.162.175.115
Deleting volume vol-04d71a810478dec0d
Cleaning up resources in VPC: vpc-fad1139d
Cleaning up security group: sg-d9280ba0
Cleaning up security group: sg-dc280ba5
Deleting security group: sg-d9280ba0
Deleting security group: sg-dc280ba5
Deleting VPC: vpc-fad1139d
Done
```


## For experts with Kubernetes and AWS

Sometimes we might need to create or manage the cluster on AWS manually with limited privileges, so here we will explain more on what’s going on with the Kubernetes setup script.

### Some Presumptions

* Instances run on Debian, the official IAM, and the filesystem is aufs instead of ext4.
* Kubernetes node use instance storage, no EBS get mounted.  Master use a persistent volume for etcd.
* Nodes are running in an Auto Scaling Group on AWS, auto-scaling itself is disabled, but if some node get terminated, it will launch another node instead.
* For networking, we use ip-per-pod model here, each pod get assigned a /24 CIDR. And the whole vpc is a /16 CIDR, No overlay network at this moment, we will add Calico solution later on.
* When you create a service with Type=LoadBalancer, Kubernetes will create and ELB, and create a security group for the ELB.
* Kube-proxy sets up two IAM roles, one for master called kubernetes-master, one for nodes called kubernetes-node.
* All AWS resources are tagged with a tag named "KubernetesCluster", with a value that is the unique cluster-id.


###Script Details

* Create an s3 bucket for binaries and scripts.
* Create two iam roles: kubernetes-master, kubernetes-node.
* Create an AWS SSH key named kubernetes-YOUR_RSA_FINGERPRINT.
* Create a vpc with 172.20.0.0/16 CIDR, and enables dns-support and dns-hostnames options in vpc settings.
* Create Internet gateway, route table, a subnet with CIDR of 172.20.0.0/24, and associate the subnet to the route table.
* Create and configure security group for master and nodes.
* Create an EBS for master, it will be attached after the master node get up.
* Launch the master with fixed ip address 172.20.0.9, and the node is initialized with Salt script, all the components get started as docker containers.
* Create an auto-scaling group, it has the min and max size, it can be changed by using aws api or console, it will auto launch the kubernetes node and configure itself, connect to master, assign an internal CIDR, and the master configures the route table with the assigned CIDR.



