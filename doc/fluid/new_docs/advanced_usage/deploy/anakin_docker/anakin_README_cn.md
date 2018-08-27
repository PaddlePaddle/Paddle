# Anakin 2.0 Docker

## 依赖软件

+ 你的操作系统上应该已经安装了docker.
+ 如果你要在docker中使用`NVIDIA GPU` 还需要安装[nvidia-docker2](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))

## 使用方法

推荐使用 `anakin_docker_build_and_run.sh` 脚本来构建和运行docker镜像，脚本的使用方法如下

```bash
Usage: anakin_docker_build_and_run.sh -p <place> -o <os> -m <Optional>

选项:

   -p     硬件的运行环境 [ NVIDIA-GPU / AMD_GPU / X86-ONLY / ARM ]
   -o     主机的操作系统类型 [ Centos / Ubuntu ]
   -m     脚本的执行模式[ Build / Run / All] 默认模式是 build and run
```

### GPU Docker

#### 构建镜像

```bash
/usr/bash anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Build
或者
chmod +x ./anakin_docker_build_and_run.sh
./anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Build
```

#### 运行 docker容器

```bash
/usr/bash anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Run
或者
chmod +x ./anakin_docker_build_and_run.sh
./anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Run
```

当docker运行成功后，你可以在./Anakin目录下找到Anakin代码

### X86 Docker

#### 构建镜像

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p X86-ONLY -o Centos -m Build

# or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p X86-ONLY -o Centos -m Build
```

#### 运行 docker容器

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p X86-ONLY -o Centos -m Run

# or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p X86-ONLY -o Centos -m Run

# or
# run docker by docker run command.
# firt find the x86 image you just built.

$docker images

# This command will list all docker images you have built.
#Then, find anakin and it's tag. finally, run x86 docker.
# for ubuntu, type anakin:x86_ubuntu16.04-x86 instead.
$docker run -it anakin:x86_centos7-x86 /bin/bash
```

当docker运行成功后，你可以在./Anakin目录下找到Anakin代码

### ARM Docer

我们目前只支持centos系统系统下的arm docker

#### 构建镜像

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p ARM -o Centos -m Build

# or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p ARM -o Centos -m Build
```

#### 运行 docker容器

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p ARM -o Centos -m Run

# or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p ARM -o Centos -m Run

# or run docker by docker run command.
# firt find the arm image you just built.

$docker images

# This command will list all docker images you have built.
#Then, find anakin and it's tag. finally, run arm docker.

$docker run -it anakin:arm_centos7-armv7 /bin/bash

```

当docker运行成功后，你可以在./Anakin目录下找到Anakin代码
