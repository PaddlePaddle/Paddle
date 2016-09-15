Build and Install
=================

## Requirement

### Dependents

- **CMake**: required for 2.8+ version
- **g++**: a recent c++ compiler supporting c++11, >= 4.6, < 5
- **BLAS library**: such as openBLAS, MKL, ATLAS
- **protobuf**: required for 2.4+ version, 3.x is not supported
- **python**: currently only 2.7 version is supported

### Optional

PaddlePaddle also support some build options, you have to install related libraries. 

- **WITH_GPU**: Compile with gpu mode
  - The GPU version works best with Cuda Toolkit 7.5 and cuDNN v5
  - Other versions Cuda Toolkit 6.5, 7.0 and cuDNN v2, v3, v4 are also supported
  - Note: to utilize cuDNN v5, Cuda Toolkit 7.5 is prerequisite and vice versa
- **WITH_DOUBLE**: Compile with double precision, otherwise use single precision 
- **WITH_GLOG**: Compile with glog, otherwise use a log implement internally
- **WITH_GFLAGS**: Compile with gflags, otherwise use a flag implement internally
- **WITH_TESTING**: Compile with gtest and run unittest for PaddlePaddle 
- **WITH_DOC**: Compile with documentation
- **WITH_SWIG_PY**: Compile with python predict api
- **WITH_STYLE_CHECK**: Style check for source code


## Building on Ubuntu14.04

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
    
  
- **GPU Dependencies(optional)**

If you need to build GPU version, the first thing you need is a machine that has GPU and CUDA installed.
And you also need to install cuDNN.

You can download CUDA toolkit and cuDNN from nvidia website:
    
```bash
https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cudnn
```
You can copy cuDNN files into the CUDA toolkit directory, such as:

```bash
sudo tar -xzf cudnn-7.5-linux-x64-v5.1.tgz -C /usr/local
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
Then you need to set LD\_LIBRARY\_PATH, CUDA\_HOME and PATH environment variables in ~/.bashrc.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
```
- **Python Dependencies(optional)**

If you want to compile PaddlePaddle with python predict api, you need to add -DWITH_SWIG_PY=ON in cmake command and install these first:

```bash
sudo apt-get install swig
```

- **Doc Dependencies(optional)**

If you want to compile PaddlePaddle with doc, you need to add -DWITH_DOC=ON in cmake command and install these first:

```bash
pip install 'sphinx>=1.4.0'
pip install sphinx_rtd_theme breathe recommonmark
sudo apt-get install doxygen 
```

### Build and Install

CMake will find dependent libraries in system default paths first. After installing some optional libraries, corresponding build option will automatically be on(such as glog, gtest and gflags). And if libraries are not found, you have to set following variables manually in cmake command(CUDNN_ROOT, ATLAS_ROOT, MKL_ROOT, OPENBLAS_ROOT).

Here are some examples of cmake command with different options:

**only cpu**

```bash
cmake -DWITH_GPU=OFF -DWITH_DOC=OFF
```

**gpu**

```bash
cmake -DWITH_GPU=ON -DWITH_DOC=OFF
```

**gpu with doc and swig**

```bash
cmake -DWITH_GPU=ON -DWITH_DOC=ON -DWITH_SWIG_PY=ON
``` 

Finally, you can download source code and build:

```bash
git clone https://github.com/baidu/Paddle paddle
cd paddle
mkdir build
cd build
# you can add build option here, such as:    
cmake -DWITH_GPU=ON -DWITH_DOC=OFF -DCMAKE_INSTALL_PREFIX=<path to install> ..
# please use sudo make install, if you want
# to install PaddlePaddle into the system
make -j `nproc` && make install
# PaddlePaddle installation path
export PATH=<path to install>/bin:$PATH
```
**Note**

And if you set WITH_SWIG_PY=ON, you have to install related python predict api at the same time:

```bash
pip install <path to install>/opt/paddle/share/wheels/*.whl
```
