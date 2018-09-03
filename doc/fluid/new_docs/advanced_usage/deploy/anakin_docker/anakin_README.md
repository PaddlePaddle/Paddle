# Anakin 2.0 Docker

## Requirement

+ You should install docker in you local os.
+ Please use [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))  build and run all `NVIDIA GPU` docker images.

## Usage

You are recommended to use `anakin_docker_build_and_run.sh` script to build and run anakin docker.

```bash
Usage: anakin_docker_build_and_run.sh -p <place> -o <os> -m <Optional>

Options:

   -p     Hardware Place where docker will running [ NVIDIA-GPU / AMD_GPU / X86-ONLY / ARM ]
   -o     Operating system docker will reside on [ Centos / Ubuntu ]
   -m     Script exe mode [ Build / Run / All] default mode is build and run
```

### GPU Docker

#### Build Image

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Build
or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Build
```

#### Run docker

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Run
or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p NVIDIA-GPU -o Centos -m Run
```

When running docker successfully, you can find Anakin source code at /Anakin directory.

### X86 Docker

#### Build Image

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p X86-ONLY -o Centos -m Build

# or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p X86-ONLY -o Centos -m Build
```

#### Run docker

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

When running docker successfully, you can find Anakin source code at /Anakin directory.

### ARM Docker

We only support arm docker based on centos for now.

#### Build Image

```bash
$/usr/bash anakin_docker_build_and_run.sh  -p ARM -o Centos -m Build

# or
$chmod +x ./anakin_docker_build_and_run.sh
$./anakin_docker_build_and_run.sh  -p ARM -o Centos -m Build
```

#### Run docker

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

When running docker successfully, you can find Anakin source code at /Anakin directory.

### NOTE

If you want to use opencv(only used in Anakin samples), please refer [run on arm](../docs/Manual/run_on_arm_en.md) to install opencv and recompile Anakin
(you just need to run 'Anakin/tools/andrid_build.sh').  You can install it before the arm docker image's building has done successfully. At last, don't forget to commit your changes to the docker image you built.
