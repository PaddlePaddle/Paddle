# Distributed Training with NCCL2 and RDMA

When doing distributed multi-GPU training, network bandwith often becomes the
bottle neck. We introduce a way to use NCCL2 to do such training job to
achieve best performace.

## Prepare Hardwares with RDMA and Multiple GPUs

I'm using two Linux servers each of them is installed with 8 GPUs and
one 100Gb RDMA card.
Base environment is:

* OS: CentOS 7.4
* RDMA device: "Mellanox Technologies MT27700 Family [ConnectX-4]"
* Kernel version: `4.4.88-1.el7.elrepo.x86_64`
* Docker version: `1.12.6`
* Docker storage driver: `overlay2`
* IP addresses: 192.168.16.30,192.168.16.34

In general, the steps including:

1. Install GPU drivers
1. Install RDMA drivers
1. Install "InfiniBand Support"
1. Use docker to run tests and make sure GPUs and RDMA can work inside
   the container.

I'll ommit section "Install GPU drivers" because we can find it easily
somewhere else.

### Install RDMA drivers

For my case, I've got two machines with device
"Mellanox Technologies MT27700 Family [ConnectX-4]" installed. The OS was
"CentOS 7.4" and I updated the kernel to version 4.4 so that docker can
work with latest overlay2 filesystem.

***NOTE: before you start, make sure you have a way to get a console
of the server other than ssh because we may need to re-configure the
network device.***

1. Go to http://www.mellanox.com/page/products_dyn?product_family=26,
   download `MLNX_OFED` software in the bottom of the page, and upload it
   onto the server.
1. Run `./mlnxofedinstall --add-kernel-support` in the software package.
1. Run `/etc/init.d/openibd restart` to make everything work, note that
   this operation may cause the network goes down if you are using this
   RDMA device as default network device and use ssh to login the server.
1. Re-configure the network interface, for example:
   `ifconfig eth2 192.168.16.30/20 up`, then add routes if needed:
   `ip route add default via 192.168.16.1 dev eth2`.
1. Do the same thing on the other node.
1. Use `ping` to test if the two nodes have typical ICMP connection.
1. Use either `udaddy` or `ib_write_bw` to test the network connection is
   ready and have the desired bandwith.

### Prepare Docker Image to Run RDMA Programs

1. Build a docker image using cuda base image like: `nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04` and install paddlepaddle whl
   package in it.
1. Start a docker container and mount GPU driver libs into it (you can
   skip this step if you are using nvidia-docker).
1. Mount RDMA dirvers and libs into the docker image (see below section),
   also `udaddy` and `ib_write_bw` if needed.
1. Mount GPU devices and RDMA devices into the container using `--device`
   or just use privileged mode `--privileged`.
1. Start the container using host network mode: `--net=host`

### RDMA Library Files Needed

Usually, `MLNX_OFED` install latest supported libs under
`/usr/lib64/mlnx_ofed/valgrind`. Other libs also needed to run RDMA programs
is listed below. These libs must be mounted into the docker container.

* Libs under `/usr/lib64/mlnx_ofed/valgrind`
  * libibcm.so
  * libibverbs.so
  * libmlx4.so
  * libmlx5.so
  * libmlx5-rdmav2.so
  * librdmacm.so
* Other libs:
  * libnl-3.so.200
  * libnl-route-3.so.200
  * libnuma.so.1

## Start to Run the Training Job

Setting NCCL environment variables to turn NCCL switches on and off:


| Env Name | Description |
| --- | --- |
| NCCL_SOCKET_IFNAME | The RDMA device, e.g. eth2 |
| NCCL_P2P_DISABLE | Set to 1 to disable P2P transfer between GPUs |
| NCCL_IB_DISABLE | Set to 1 to disable using RDMA |
| NCCL_IB_CUDA_SUPPORT | Set to 1 to enable GPU Direct if supported |
| NCCL_DEBUG | Set debug level: VERSION, WARN, INFO |

My two servers are: `192.168.16.30,192.168.16.34`, On node 1, Run :

```bash
PADDLE_TRAINER_ID=0 PADDLE_PORT=48372 PADDLE_WORKERS=192.168.16.30,192.168.16.34 POD_IP=192.168.16.30 stdbuf -oL python vgg16.py
```

On node 2, Run:

```bash
PADDLE_TRAINER_ID=1 PADDLE_PORT=48372 PADDLE_WORKERS=192.168.16.30,192.168.16.34 POD_IP=192.168.16.34 stdbuf -oL python vgg16.py
```
