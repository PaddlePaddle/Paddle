# Cluster bootstrapping tool survey
## Abstract
In order to bring up a cluster from bare metal machine to a fully functional kubernetes cluster for Paddlepaddle to run, we need to utilize some tools. Here we are going to compare [Sextant](https://github.com/k8sp/sextant) and [Tectonic installer](https://github.com/coreos/tectonic-installer)

## Basic assumptions
Here are some basic assumptions before we move on to  details
1. You are an administrator of a bare metal machine cluster, which means:
  * you have full control to each of the machines.
  * you have full control to the network which machines are connected to.
2. Machines can be booted from network with PEX or iPXE
3. You understand the [general procedure to bring up a cluster](#appendix-general-procedure-to-bring-up-a-cluster)

if your cluster is able to mark above items with checkmarks, then keep reading.

## Comparing Sextant and Tectonic installer
### Sextant
Sextant is an end2end solution to bring up a bare metal cluster to a fully functional k8s cluster, it integrates DHCP, name service, PEX, cloud-config-service, docker registry services altogether. 

#### Pros
1. End2End: basically all admin need to do is to config the cluster.yaml and power on the cluster.
2. Offline cluster configuration: Sextant has 2 phases during working with it, config time and deploy time. when admin is configuring, it requires admin's machine has internet connectivity, which will download some images, etc. But in deploy time, it's completely OK to go offline since all dependencies are ready during config time.
3. docker registry integrated.
4. GPU machine took care of.

### Cons
1. k8s API server is not deployed with high availability in considering by default.
2. No grouping support.
3. No API interface, a one-off service.


### Tectonic installer
First of all, Tectonic is not free, it requires coreos.com account as a step of installation, and free user can only create less than 10 nodes.

Tectonic is a suite of software which wraps around k8s and providing more utility regarding dev ops, ie, 
Tectonic installer as it's named, it installs Tectonic to a bare metal cluster which means it's not totally an equivalent of Sextant. At the "booting a cluster" part, it mostly utilizes [Matchbox](https://github.com/coreos/matchbox), which is a general cluster bootstrapper.

Matchbox's Approach is similar to Sexstant.

### Pros
1. supports grouping machines.
2. supports running provisioning service in rtk. (not a big deal though).
3. supports http/gRPC API interface.
4. supports multi-template.

### Cons
1. Not an e2e solution to bring up a cluster, need a lot of extra work and other software.
2. [Not fully supporting](https://github.com/coreos/matchbox/issues/550) centOS deployment yet.

## Conclusion
Sextant is a better solution overall for paddle cloud deploying to a bare metal cluster. It would be great if Sextant can also 1) deploy k8s api server with high availability by default; 2) not designed as a one-off service.



## Appendix: General procedure to bring up a cluster
It's physically impossible for a cluster admin to manually install OS and applications into cluster nodes one by one, here is what an admin would do in cloud industry:
1. setup a bootstrap machine with static IP in the cluster, which has following services:
  * DHCP: assigns ip address for rest of the nodes.
  * name service: to map node name to a IP
  * PXE related services: the booting related info will be delivered to newly booted machines as their IP is assigned via DHCP service, PXE service will provide further booting and installing info and image with TFTP and http protocol. 
  * cluster config service: this is for providing cluster node with OS config via http
  * optional docker registry: a built-in docker registry makes the whole cluster independent from connecting internet, and speeds up software distribution.
2. New node powers on, it will
  * broadcast the request for an IP address
  * DHCP server assigns the IP address, and deliver the PXE booting related info to the node.
  * cluster node will request config files with booting info delivered with DHCP via the TFTP service, and in most of the cases, the config file will point to a http service for the booting image.
  * Since PXE is configured with initrd, it will utilize the cloud config service and do further installations like coreOS or K8s installations.
  * then restart the node.

For further understanding, following 2 links from Matchbox are some good readings:
* [Machine lifecycle](https://github.com/coreos/matchbox/blob/master/Documentation/machine-lifecycle.md)
* [PXE booting](https://github.com/coreos/matchbox/blob/master/Documentation/network-booting.md)
