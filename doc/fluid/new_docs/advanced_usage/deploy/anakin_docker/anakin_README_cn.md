# Anakin 2.0 Docker
---

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

### X86 Docker

> Not support yet

### ARM Docer

> Not support yet
