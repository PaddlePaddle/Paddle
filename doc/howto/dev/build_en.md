# Build using Docker

## What Developers Need

To contribute to PaddlePaddle, you need

1. A computer -- Linux, BSD, Windows, MacOS, and
1. Docker.

Nothing else.  Not even Python and GCC, because you can install all build tools into a Docker image.  We run all the tools by running this image.

## General Process

1. Retrieve source code.

   ```bash
   git clone https://github.com/paddlepaddle/paddle
   ```

2. Install build tools into a Docker image.

   ```bash
   cd paddle; docker build -t paddle:dev .
   ```

   Please be aware of the `.` at the end of the command, which refers to the [`./Dockerfile` file](https://github.com/PaddlePaddle/Paddle/blob/develop/Dockerfile).  `docker build` follows instructions in this file to create a Docker image named `paddle:dev`, and installs building tools into it.

3. Build from source.

   This following command starts a Docker container that executes the Docker image `paddle:dev`, mapping the current directory to `/paddle/` in the container, and runs the default entry-point [`build.sh`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/docker/build.sh) as specified in the Dockefile.  `build.sh` invokes `cmake` and `make` to build PaddlePaddle source code, which had been mapped to `/paddle`, and writes outputs to `/paddle/build`, which maps to `build` in the current source directory on the computer.

   ```bash
   docker run -v $PWD:/paddle paddle:dev
   ```

   Above command builds a CUDA-enabled version.  If we want to build a CPU-only version, we can type

   ```bash
   docker run -e WITH_GPU=OFF -v $PWD:/paddle paddle:dev
   ```

4. Run unit tests.

   To run all unit tests using the first GPU of a node:

   ```bash
   NV_GPU=0 nvidia-docker run -v $PWD:/paddle paddle:dev bash -c "cd /paddle/build; ctest"
   ```

   If we used `WITH_GPU=OFF` at build time, it generates only CPU-based unit tests, and we don't need nvidia-docker to run them.  We can just run

   ```bash
   docker run -v $PWD:/paddle paddle:dev bash -c "cd /paddle/build; ctest"
   ```

   Sometimes we want to run a specific unit test, say `memory_test`, we can run

   ```bash
   nvidia-docker run -v $PWD:/paddle paddle:dev bash -c "cd /paddle/build; ctest -V -R memory_test"
   ```

5. Clean Build.

   Sometimes, we might want to clean all thirt-party dependents and built binaries.  To do so, just

   ```bash
   rm -rf build
   ```

## Docker, Or Not?

- What is Docker?

  If you haven't heard of it, consider it something like Python's virtualenv.

- Docker or virtual machine?

  Some people compare Docker with VMs, but Docker doesn't virtualize any hardware nor running a guest OS, which means there is no compromise on the performance.

- Why Docker?

  Using a Docker image of build tools standardizes the building environment, which makes it easier for others to reproduce your problems and to help.

  Also, some build tools don't run on Windows or Mac or BSD, but Docker runs almost everywhere, so developers can use whatever computer they want.

- Can I choose not to use Docker?

  Sure, you don't have to install build tools into a Docker image; instead, you can install them in your local computer.  This document exists because Docker would make the development way easier.

- How difficult is it to learn Docker?

    It takes you ten minutes to read [an introductory article](https://docs.docker.com/get-started) and saves you more than one hour to install all required build tools, configure them, especially when new versions of PaddlePaddle require some new tools.  Not even to mention the time saved when other people trying to reproduce the issue you have.

- Can I use my favorite IDE?

  Yes, of course.  The source code resides on your local computer, and you can edit it using whatever editor you like.

  Many PaddlePaddle developers are using Emacs.  They add the following few lines into their `~/.emacs` configure file:

  ```emacs
  (global-set-key "\C-cc" 'compile)
  (setq compile-command
   "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev")
  ```

  so they could type `Ctrl-C` and `c` to build PaddlePaddle from source.

- Does Docker do parallel building?

  Our building Docker image runs a [Bash script](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/docker/build.sh), which calls `make -j$(nproc)` to starts as many processes as the number of your CPU cores.

## Some Gotchas

- Docker requires sudo

  An owner of a computer has the administrative privilege, a.k.a., sudo, and Docker requires this privilege to work properly.  If you use a shared computer for development, please ask the administrator to install and configure Docker.  We will do our best to support rkt, another container technology that doesn't require sudo.

- Docker on Windows/MacOS builds slowly

  On Windows and MacOS, Docker containers run in a Linux VM.  You might want to give this VM some more memory and CPUs so to make the building efficient.  Please refer to [this issue](https://github.com/PaddlePaddle/Paddle/issues/627) for details.

- Not enough disk space

  Examples in this article uses option `--rm` with the `docker run` command.  This option ensures that stopped containers do not exist on hard disks.  We can use `docker ps -a` to list all containers, including stopped.  Sometimes `docker build` generates some intermediate dangling images, which also take disk space.  To clean them, please refer to [this article](https://zaiste.net/posts/removing_docker_containers/).
