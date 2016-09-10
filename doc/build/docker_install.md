Docker installation guide
====================
PaddlePaddle provides some pre-compiled binary, including Docker images, ubuntu deb packages. It is welcomed to contributed more installation package of different linux distribution (such as ubuntu, centos, debian, gentoo and so on). We recommend to use Docker images to deploy PaddlePaddle.
## Docker installation

Docker is a tool designed to make it easier to create, deploy, and run applications by using containers.

### PaddlePaddle Docker images
There are six Docker images:

- paddledev/paddle:cpu-latest: PaddlePaddle CPU binary image.
- paddledev/paddle:gpu-latest: PaddlePaddle GPU binary image.
- paddledev/paddle:cpu-devel-latest: PaddlePaddle CPU binary image plus source code.
- paddledev/paddle:gpu-devel-latest: PaddlePaddle GPU binary image plus source code.
- paddledev/paddle:cpu-demo-latest: PaddlePaddle CPU binary image plus source code and demo
- paddledev/paddle:gpu-demo-latest: PaddlePaddle GPU binary image plus source code and demo

Tags with latest will be replaced by a released version. 

### Download and Run Docker images

You have to install Docker in your machine which has linux kernel version 3.10+ first. You can refer to the official guide https://docs.docker.com/engine/installation/ for further information.

You can use ```docker pull ```to download images first, or just launch a container with ```docker run```:
```bash
docker run -it paddledev/paddle:cpu-latest
```

If you want to launch container with GPU support, you need to set some environment variables at the same time:

```bash
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
docker run -it paddledev/paddle:gpu-latest
``` 

### Notice

#### Performance

Since Docker is based on the lightweight virtual containers, the CPU computing performance maintains well. And GPU driver and equipments are all mapped to the container, so the GPU computing performance would not be seriously affected.

If you use high performance nic, such as RDMA(RoCE 40GbE or IB 56GbE), Ethernet(10GbE), it is recommended to use config "-net = host".




#### Remote access
If you want to enable ssh access background, you need to build an image by yourself. Please refer to official guide https://docs.docker.com/engine/reference/builder/ for further information.

Following is a simple Dockerfile with ssh:
```bash
FROM paddledev/paddle

MAINTAINER PaddlePaddle dev team <paddle-dev@baidu.com>

RUN apt-get update
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd

RUN sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

EXPOSE 22

CMD    ["/usr/sbin/sshd", "-D"]
```

Then you can build an image with Dockerfile and launch a container:

```bash
# cd into Dockerfile directory
docker build . -t paddle_ssh
# run container, and map host machine port 8022 to container port 22
docker run -d -p 8022:22 --name paddle_ssh_machine paddle_ssh
```
Now, you can ssh on port 8022 to access the container, username is root, password is also root:

```bash
ssh -p 8022 root@YOUR_HOST_MACHINE
```


You can stop and delete the container as following:
```bash
# stop
docker stop paddle_ssh_machine
# delete
docker rm paddle_ssh_machine
```
