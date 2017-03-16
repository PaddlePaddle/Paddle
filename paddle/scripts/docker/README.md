We need to complete the initial draft https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/docker/README.md.

I am recording some ideas here, and we should file a PR later.

## Current Status

Currently, we have four sets of Dockefiles:

1. Kubernetes examples:

   ```
   doc/howto/usage/k8s/src/Dockerfile -- based on released image but add start.sh
   doc/howto/usage/k8s/src/k8s_data/Dockerfile -- contains only get_data.sh
   doc/howto/usage/k8s/src/k8s_train/Dockerfile -- this duplicates with the first one.
   ```

1. Generate .deb packages:

   ```
   paddle/scripts/deb/build_scripts/Dockerfile -- significantly overlaps with the `docker` directory
   ```

1. In the `docker` directory:

   ```
   paddle/scripts/docker/Dockerfile
   paddle/scripts/docker/Dockerfile.gpu
   ```

1. Document building

   ```
   paddle/scripts/tools/build_docs/Dockerfile -- a subset of above two sets.
   ```

## Goal

We want two Docker images for each version of PaddlePaddle:

1. `paddle:<version>-dev`

   This a development image contains only the development tools.  This standardizes the building tools and procedure.  Users include:

   - developers -- no longer need to install development tools on the host, and can build their current work on the host (development computer).
   - release engineers -- use this to build the official release from certain branch/tag on Github.com.
   - document writers / Website developers -- Our documents are in the source repo in the form of .md/.rst files and comments in source code.  We need tools to extract the information, typeset, and generate Web pages.

   So the development image must contain not only source code building tools, but also documentation tools:

   - gcc/clang
   - nvcc
   - Python
   - sphinx
   - woboq
   - sshd

   where `sshd` makes it easy for developers to have multiple terminals connecting into the container.

1. `paddle:<version>`

   This is the production image, generated using the development image. This image might have multiple variants:

   - GPU/AVX   `paddle:<version>-gpu`
   - GPU/no-AVX  `paddle:<version>-gpu-noavx`
   - no-GPU/AVX  `paddle:<version>`
   - no-GPU/no-AVX  `paddle:<version>-noavx`

   We'd like to give users choices of GPU and no-GPU, because the GPU version image is much larger than then the no-GPU version.

   We'd like to give users choices of AVX and no-AVX, because some cloud providers don't provide AVX-enabled VMs.

## Dockerfile

To realize above goals, we need only one Dockerfile for the development image.  We can put it in the root source directory.

Let us go over our daily development procedure to show how developers can use this file.

1. Check out the source code

   ```bash
   git clone https://github.com/PaddlePaddle/Paddle paddle
   ```

1. Do something

   ```bash
   cd paddle
   git checkout -b my_work
   Edit some files
   ```

1. Build/update the development image (if not yet)

   ```bash
   docker build -t paddle:dev . # Suppose that the Dockerfile is in the root source directory.
   ```

1. Build the source code

   ```bash
   docker run -v $PWD:/paddle -e "GPU=OFF" -e "AVX=ON" -e "TEST=ON" paddle:dev
   ```

   This command maps the source directory on the host into `/paddle` in the container.

   Please be aware that the default entrypoint of `paddle:dev` is a shell script file `build.sh`, which builds the source code, and outputs to `/paddle/build` in the container, which is actually `$PWD/build` on the host.

   `build.sh` doesn't only build binaries, but also generates a `$PWD/build/Dockerfile` file, which can be used to build the production image.  We will talk about it later.

1. Run on the host (Not recommended)

   If the host computer happens to have all dependent libraries and Python runtimes installed, we can now run/test the built program.  But the recommended way is to running in a production image.

1. Run in the development container

   `build.sh` generates binary files and invokes `make install`.  So we can run the built program within the development container.  This is convenient for developers.

1. Build a production image

    On the host, we can use the `$PWD/build/Dockerfile` to generate a production image.

   ```bash
   docker build -t paddle --build-arg "BOOK=ON" -f build/Dockerfile .
   ```

1. Run the Paddle Book

   Once we have the production image, we can run [Paddle Book](http://book.paddlepaddle.org/) chapters in Jupyter Notebooks (if we chose to build them)

   ```bash
   docker run -it paddle
   ```

   Note that the default entrypoint of the production image starts Jupyter server, if we chose to build Paddle Book.

1. Run on Kubernetes

   We can push the production image to a DockerHub server, so developers can run distributed training jobs on the Kuberentes cluster:

   ```bash
   docker tag paddle me/paddle
   docker push
   kubectl ...
   ```

   For end users, we will provide more convinient tools to run distributed jobs.
