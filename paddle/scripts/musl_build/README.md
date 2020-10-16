Paddle for Linux-musl Usage Guide
===========================================

# introduction
Paddle can be built for linux-musl such as alpine, and be used in libos-liked SGX TEE environment. Currently supported commericial product TEE Scone, and community maintanced TEE Occlum. We also working on to support open source TEE Graphene.


# build automaticly
1. clone paddle source from github
   
```bash
git clone https://github.com/PaddlePaddle/Paddle.git
```

2. setup build directory

```bash
# enter paddle directory
cd  ./Paddle

# create and enter building directory
mkdir -p build && cd build
```

3. build docker for compiling. use environment HTTP_PROXY/HTTPS_PROXY for proxy setup.

```bash
# setup proxy address
export HTTP_PROXY='http://127.0.0.1:8080'
export HTTPS_PROXY='https://127.0.0.1:8080'

# invoke build script
../paddle/scripts/musl_build/build_docker.sh
```

4. compile paddle in previous built docker. proxy setup method is same as previous step.
output wheel package will save to "dist" directory.

```bash
# setup proxy addresss
export HTTP_PROXY='http://127.0.0.1:8080'
export HTTPS_PROXY='https://127.0.0.1:8080'

# invoke build paddle script
../paddle/scripts/musl_build/build_paddle.sh

# find output wheel package
ls dist/*.whl
```

# build paddle manually  

1. start up the building docker, and enter the shell in the container
```bash
# checkout paddle source code
git clone https://github.com/PaddlePaddle/Paddle.git

# entery paddle directory
cd ./Paddle

# build docker image
../paddle/scripts/musl_build/build_docker.sh

# enter the container interactive shell
BUILD_AUTO=0 ../paddle/scripts/musl_build/build_paddle.sh
```

2. Type commands to compile source manually
```sh
# compile paddle by commands
# paddle is mount to /paddle directory
# working directory is /root
mkdir build && cd build

# install python requirement
pip install -r /paddle/python/requirements.txt

# configure project with cmake
cmake /paddle -DWITH_MUSL=ON DWITH_CRYPTO=OFF -DWITH_MKL=OFF -DWITH_GPU=OFF -DWITH_TESTING=OFF

# run the make to build project
make
```

# files
- build_docker.sh: docker building script
- build_paddle.sh: paddle building script
- build_inside.sh: build_paddle.sh will invoke this script inside the docker for compiling.
- config.sh: build config script for configure compiling option setting.
- Dockerfile: build docker defination file.
