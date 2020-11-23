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
# setup proxy address, when the speed of internet is not good.
# export HTTP_PROXY='http://127.0.0.1:8080'
# export HTTPS_PROXY='https://127.0.0.1:8080'

# invoke build script
../paddle/scripts/musl_build/build_docker.sh
```

4. compile paddle in previous built docker. proxy setup method is same as previous step.
output wheel package will save to "dist" directory.

```bash
# setup proxy addresss, when the speed of internet is not good.
# export HTTP_PROXY='http://127.0.0.1:8080'
# export HTTPS_PROXY='https://127.0.0.1:8080'

# invoke build paddle script
# all arguments, such as -j8 optinal, is past to make procedure.
../paddle/scripts/musl_build/build_paddle.sh -j8

# find output wheel package
ls dist/*.whl
```

# build manually  

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
cmake -DWITH_MUSL=ON DWITH_CRYPTO=OFF -DWITH_MKL=OFF -DWITH_GPU=OFF -DWITH_TESTING=OFF /paddle

# run the make to build project.
# the argument -j8 is optional to accelerate compiling.
make -j8
```

# scripts
1. build_docker.sh
   
    compiling docker building script
    
    environment arguments:

   - WITH_PRUNE_DAYS: prune old docker images, with days limitation.
   - WITH_REBUILD: force to rebuild the image, default=0.
   - WITH_REQUIREMENT: build with the python requirements, default=1.
   - WITH_UT_REQUIREMENT: build with the unit test requirements, default=0.
   - WITH_PIP_INDEX: use custom pip index when pip install packages.
   - ONLY_NAME: only print the docker name, and exit.
   - HTTP_PROXY: use http proxy
   - HTTPS_PROXY: use https proxy

2. build_paddle.sh
   
    paddle building script, accept follow environment variables as arguments:

    - BUILD_AUTO: build the paddle automatically, save output wheel package to ./output directory, default=1.
    
    - HTTP_PROXY: use http proxy
    - HTTPS_PROXY: use https proxy


# files
- build_docker.sh: docker building script
- build_paddle.sh: paddle building script
- build_inside.sh: build_paddle.sh will invoke this script inside the docker for compiling.
- config.sh: build config script for configure compiling option setting.
- Dockerfile: build docker defination file.
