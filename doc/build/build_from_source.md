Installing from Sources
==========================

* [1. Download and Setup](#download)
* [2. Requirements](#requirements)
* [3. Build on Ubuntu](#ubuntu)

## <span id="download">Download and Setup</span> 
You can download PaddlePaddle from the [github source](https://github.com/gangliao/Paddle).

```bash
git clone https://github.com/baidu/Paddle paddle
cd paddle
```

## <span id="requirements">Requirements</span>

To compile the source code, your computer must be equipped with GCC >=4.6 or Clang compiler.
### Dependencies

- **CMake**: version >= 2.8
- **BLAS**: MKL, OpenBlas or ATLAS
- **protobuf**: version >= 2.4, **Note: 3.x is not supported**
- **python**: only python 2.7 is supported currently

### Options

PaddlePaddle supports some build options. To enable it, first you need to install the related libraries. 

<html>
<table> 
<thead>
<tr>
<th scope="col" class="left">Optional</th>
<th scope="col" class="left">Description</th>
</tr>
</thead>
<tbody>
<tr><td class="left">WITH_GPU</td><td class="left">Compile with GPU mode.</td></tr>
<tr><td class="left">WITH_DOUBLE</td><td class="left">Compile with double precision floating-point, default: single precision.</td></tr>
<tr><td class="left">WITH_GLOG</td><td class="left">Compile with glog. If not found, default: an internal log implementation.</td></tr>
<tr><td class="left">WITH_GFLAGS</td><td class="left">Compile with gflags. If not found, default: an internal flag implementation.</td></tr>
<tr><td class="left">WITH_TESTING</td><td class="left">Compile with gtest for PaddlePaddle's unit testing.</td></tr>
<tr><td class="left">WITH_DOC</td><td class="left">	Compile to generate PaddlePaddle's docs, default: disabled (OFF).</td></tr>
<tr><td class="left">WITH_SWIG_PY</td><td class="left">Compile with python predict API, default: disabled (OFF).</td></tr>
<tr><td class="left">WITH_STYLE_CHECK</td><td class="left">Compile with code style check, default: enabled (ON).</td></tr>
</tbody>
</table>
</html>

**Note:**
  - The GPU version works best with Cuda Toolkit 7.5 and cuDNN v5.
  - Other versions like Cuda Toolkit 6.5, 7.0, 8.0 and cuDNN v2, v3, v4 are also supported.
  - **To utilize cuDNN v5, Cuda Toolkit 7.5 is prerequisite and vice versa.**

As a simple example, consider the following:  

1. **Python Dependencies(optional)**
  
    To compile PaddlePaddle with python predict API, make sure swig installed and set `-DWITH_SWIG_PY=ON` as follows:

    ```bash
    # install swig on ubuntu
    sudo apt-get install swig
    # install swig on Mac OS X
    brew install swig

    # active swig in cmake
    cmake .. -DWITH_SWIG_PY=ON
    ```

2. **Doc Dependencies(optional)**

    To generate PaddlePaddle's documentation, install dependencies and set `-DWITH_DOC=ON` as follows:

    ```bash
    pip install 'sphinx>=1.4.0'
    pip install sphinx_rtd_theme breathe recommonmark

    # install doxygen on Ubuntu
    sudo apt-get install doxygen 
    # install doxygen on Mac OS X
    brew install doxygen

    # active docs in cmake
    cmake .. -DWITH_DOC=ON`
    ```

## <span id="ubuntu">Build on Ubuntu 14.04</span>

### Install Dependencies

- **CPU Dependencies**

    ```bash
    # necessary
    sudo apt-get update
    sudo apt-get install -y g++ make cmake build-essential libatlas-base-dev python python-pip libpython-dev m4 libprotobuf-dev protobuf-compiler python-protobuf python-numpy git
    # optional
    sudo apt-get install libgoogle-glog-dev
    sudo apt-get install libgflags-dev
    sudo apt-get install libgtest-dev
    sudo pip install wheel
    pushd /usr/src/gtest
    cmake .
    make
    sudo cp *.a /usr/lib
    popd
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
cmake ..
```

CMake first check PaddlePaddle's dependencies in system default path. After installing some optional
libraries, corresponding build option will be set automatically (for instance, glog, gtest and gflags).
If still not found, you can manually set it based on CMake error information from your screen.

As a simple example, consider the following:

- **Only CPU**

  ```bash
  cmake  .. -DWITH_GPU=OFF
  ```
- **GPU**

  ```bash
  cmake .. -DWITH_GPU=ON
  ```

- **GPU with doc and swig**

  ```bash
  cmake .. -DWITH_GPU=ON -DWITH_DOC=ON -DWITH_SWIG_PY=ON
  ``` 

Finally, you can build PaddlePaddle:

```bash
# you can add build option here, such as:    
cmake .. -DWITH_GPU=ON -DCMAKE_INSTALL_PREFIX=<path to install>
# please use sudo make install, if you want to install PaddlePaddle into the system
make -j `nproc` && make install
# set PaddlePaddle installation path in ~/.bashrc
export PATH=<path to install>/bin:$PATH
```

**Note:**

If you set `WITH_SWIG_PY=ON`, related python dependencies also need to be installed.
Otherwise, PaddlePaddle will automatically install python dependencies
at first time when user run paddle commands, such as `paddle version`, `paddle train`.
It may require sudo privileges:

```bash
# you can run
sudo pip install <path to install>/opt/paddle/share/wheels/*.whl
# or just run 
sudo paddle version
```
