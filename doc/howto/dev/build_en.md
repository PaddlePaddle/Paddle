# Build PaddlePaddle from Source Code and Run Unit Test

## What Developers Need

To contribute to PaddlePaddle, you need

1. A computer -- Linux, BSD, Windows, MacOS, and
1. Docker.

Nothing else.  Not even Python and GCC, because you can install all build tools into a Docker image.

## General Process

1. Retrieve source code.

   ```bash
   git clone https://github.com/paddlepaddle/paddle
   ```

2. Install build tools.

   ```bash
   cd paddle; docker build -t paddle:dev .
   ```

3. Build from source.

   ```bash
   docker run -v $PWD:/paddle paddle:dev
   ```

   This builds a CUDA-enabled version and writes all binary outputs to directory `./build` of the local computer, other than the Docker container.  If we want to build only the CPU part, we can type

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

## Docker, Or Not?

- What is Docker?

  If you haven't heard of it, consider it something like Python's virtualenv.

- Docker or virtual machine?

  Some people compare Docker with VMs, but Docker doesn't virtualize any hardware, and it doesn't run a guest OS.

- Why Docker?

  Using a Docker image of build tools standardize the building environment, and easier for others to reproduce your problem, if there is any, and help.

  Also, some build tools don't run on Windows or Mac or BSD, but Docker runs almost everywhere, so developers can use whatever computer they want.

- Can I don't use Docker?

  Sure, you don't have to install build tools into a Docker image; instead, you can install them onto your local computer.  This document exists because Docker would make the development way easier.

- How difficult is it to learn Docker?

  It takes you ten minutes to read https://docs.docker.com/get-started/ and saves you more than one hour to install all required build tools, configure them, and upgrade them when new versions of PaddlePaddle require some new tools.

- Docker requires sudo

  An owner of a computer has the administrative privilege, a.k.a., sudo.  If you use a shared computer for development, please ask the administrator to install and configure Docker.  We will do our best to support rkt, another container technology that doesn't require sudo.

- Can I use my favorite IDE?

  Yes, of course.  The source code resides on your local computer, and you can edit it using whatever editor you like.

  Many PaddlePaddle developers are using Emacs.  They add the following few lines into their `~/.emacs` configure file:

  ```emacs
  (global-set-key "\C-cc" 'compile)
  (setq compile-command
   "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev")
  ```

  so they could type `Ctrl-C` and `c` to build PaddlePaddle from source.

- How many parallel building processes does the Docker container run?

  Our building Docker image runs a Bash script https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/docker/build.sh, which calls `make -j$(nproc)` to starts as many processes as the number of your processors.
