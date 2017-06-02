Installing from Sources
==========================

* [1. Download and Setup](#download)
* [2. Requirements](#requirements)
* [3. Build on Ubuntu](#ubuntu)
* [4. Build on Centos](#centos)


## <span id="download">Download and Setup</span> 
You can download PaddlePaddle from the [github source](https://github.com/PaddlePaddle/Paddle).

```bash
git clone https://github.com/PaddlePaddle/Paddle paddle
cd paddle
```
## <span id="requirements">Requirements</span>

To compile the source code, your computer must be equipped with the following dependencies.

- **Compiler**: GCC >= 4.8 or Clang >= 3.3 (AppleClang >= 5.1) and gfortran compiler
- **CMake**: CMake >= 3.0 (at least CMake 3.4 on Mac OS X)
- **BLAS**: MKL, OpenBlas or ATLAS
- **Python**: only support Python 2.7

**Note:** For CUDA 7.0 and CUDA 7.5, GCC 5.0 and up are not supported!
For CUDA 8.0, GCC versions later than 5.3 are not supported!

### Options

PaddlePaddle supports some build options. 

<html>
<table> 
<thead>
<tr>
<th scope="col" class="left">Optional</th>
<th scope="col" class="left">Description</th>
</tr>
</thead>
<tbody>
<tr><td class="left">WITH_GPU</td><td class="left">Compile PaddlePaddle with NVIDIA GPU</td></tr>
<tr><td class="left">WITH_AVX</td><td class="left">Compile PaddlePaddle with AVX intrinsics</td></tr>
<tr><td class="left">WITH_DSO</td><td class="left">Compile PaddlePaddle with dynamic linked CUDA</td></tr>
<tr><td class="left">WITH_TESTING</td><td class="left">Compile PaddlePaddle with unit testing</td></tr>
<tr><td class="left">WITH_SWIG_PY</td><td class="left">Compile PaddlePaddle with inference api</td></tr>
<tr><td class="left">WITH_STYLE_CHECK</td><td class="left">Compile PaddlePaddle with style check</td></tr>
<tr><td class="left">WITH_PYTHON</td><td class="left">Compile PaddlePaddle with python interpreter</td></tr>
<tr><td class="left">WITH_DOUBLE</td><td class="left">Compile PaddlePaddle with double precision</td></tr>
<tr><td class="left">WITH_RDMA</td><td class="left">Compile PaddlePaddle with RDMA support</td></tr>
<tr><td class="left">WITH_TIMER</td><td class="left">Compile PaddlePaddle with stats timer</td></tr>
<tr><td class="left">WITH_PROFILER</td><td class="left">Compile PaddlePaddle with GPU profiler</td></tr>
<tr><td class="left">WITH_DOC</td><td class="left">Compile PaddlePaddle with documentation</td></tr>
<tr><td class="left">WITH_COVERAGE</td><td class="left">Compile PaddlePaddle with code coverage</td></tr>
<tr><td class="left">COVERALLS_UPLOAD</td><td class="left">Package code coverage data to coveralls</td></tr>
<tr><td class="left">ON_TRAVIS</td><td class="left">Exclude special unit test on Travis CI</td></tr>
</tbody>
</table>
</html>

**Note:**
  - The GPU version works best with Cuda Toolkit 8.0 and cuDNN v5.
  - Other versions like Cuda Toolkit 7.0, 7.5 and cuDNN v3, v4 are also supported.
  - **To utilize cuDNN v5, Cuda Toolkit 7.5 is prerequisite and vice versa.**

As a simple example, consider the following:  

1. **BLAS Dependencies(optional)**
  
    CMake will search BLAS libraries from system. If not found, OpenBLAS will be downloaded, built and installed automatically.
    To utilize preinstalled BLAS， you can simply specify MKL, OpenBLAS or ATLAS via `MKL_ROOT`, `OPENBLAS_ROOT` or `ATLAS_ROOT`.

    ```bash
    # specify MKL
    cmake .. -DMKL_ROOT=<mkl_path>
    # or specify OpenBLAS
    cmake .. -DOPENBLAS_ROOT=<openblas_path>
    ```

2. **Doc Dependencies(optional)**

    To generate PaddlePaddle's documentation, install dependencies and set `-DWITH_DOC=ON` as follows:

    ```bash
    pip install 'sphinx>=1.4.0'
    pip install sphinx_rtd_theme recommonmark

    # install doxygen on Ubuntu
    sudo apt-get install doxygen 
    # install doxygen on Mac OS X
    brew install doxygen

    # active docs in cmake
    cmake .. -DWITH_DOC=ON`
    ```

## <span id="ubuntu">Build on Ubuntu 14.04</span>

### Install Dependencies

- **Paddle Dependencies**

    ```bash
    # necessary
    sudo apt-get update
    sudo apt-get install -y git curl gcc g++ gfortran make build-essential automake
    sudo apt-get install -y python python-pip python-numpy libpython-dev bison
    sudo pip install 'protobuf==3.1.0.post1'

    # install cmake 3.4
    curl -sSL https://cmake.org/files/v3.4/cmake-3.4.1.tar.gz | tar -xz && \
        cd cmake-3.4.1 && ./bootstrap && make -j4 && sudo make install && \
        cd .. && rm -rf cmake-3.4.1
    ```

- **GPU Dependencies (optional)**

    To build GPU version, you will need the following installed:

        1. a CUDA-capable GPU
        2. A supported version of Linux with a gcc compiler and toolchain
        3. NVIDIA CUDA Toolkit (available at http://developer.nvidia.com/cuda-downloads)
        4. NVIDIA cuDNN Library (availabel at https://developer.nvidia.com/cudnn)

    The CUDA development environment relies on tight integration with the host development environment,
    including the host compiler and C runtime libraries, and is therefore only supported on
    distribution versions that have been qualified for this CUDA Toolkit release.
        
    After downloading cuDNN library, issue the following commands:

    ```bash
    sudo tar -xzf cudnn-7.5-linux-x64-v5.1.tgz -C /usr/local
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    ```
    Then you need to set LD\_LIBRARY\_PATH, PATH environment variables in ~/.bashrc.

    ```bash
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    ```

### Build and Install

As usual, the best option is to create build folder under paddle project directory.

```bash
mkdir build && cd build
``` 

Finally, you can build and install PaddlePaddle:

```bash
# you can add build option here, such as:    
cmake .. -DCMAKE_INSTALL_PREFIX=<path to install>
# please use sudo make install, if you want to install PaddlePaddle into the system
make -j `nproc` && make install
# set PaddlePaddle installation path in ~/.bashrc
export PATH=<path to install>/bin:$PATH
# install PaddlePaddle Python modules.
sudo pip install <path to install>/opt/paddle/share/wheels/*.whl
```
## <span id="centos">Build on Centos 7</span>

### Install Dependencies

- **CPU Dependencies**

    ```bash
    # necessary
    sudo yum update
    sudo yum install -y epel-release
    sudo yum install -y make cmake3 python-devel python-pip gcc-gfortran swig git
    sudo pip install wheel numpy
    sudo pip install 'protobuf>=3.0.0'
    ```
  
- **GPU Dependencies (optional)**

    To build GPU version, you will need the following installed:

        1. a CUDA-capable GPU
        2. A supported version of Linux with a gcc compiler and toolchain
        3. NVIDIA CUDA Toolkit (available at http://developer.nvidia.com/cuda-downloads)
        4. NVIDIA cuDNN Library (availabel at https://developer.nvidia.com/cudnn)

    The CUDA development environment relies on tight integration with the host development environment,
    including the host compiler and C runtime libraries, and is therefore only supported on
    distribution versions that have been qualified for this CUDA Toolkit release.
        
    After downloading cuDNN library, issue the following commands:

    ```bash
    sudo tar -xzf cudnn-7.5-linux-x64-v5.1.tgz -C /usr/local
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    ```
    Then you need to set LD\_LIBRARY\_PATH, PATH environment variables in ~/.bashrc.

    ```bash
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    ```

### Build and Install

As usual, the best option is to create build folder under paddle project directory.

```bash
mkdir build && cd build
``` 

Finally, you can build and install PaddlePaddle:

```bash
# you can add build option here, such as:    
cmake3 .. -DCMAKE_INSTALL_PREFIX=<path to install>
# please use sudo make install, if you want to install PaddlePaddle into the system
make -j `nproc` && make install
# set PaddlePaddle installation path in ~/.bashrc
export PATH=<path to install>/bin:$PATH
# install PaddlePaddle Python modules.
sudo pip install <path to install>/opt/paddle/share/wheels/*.whl
```
