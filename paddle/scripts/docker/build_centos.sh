#!/bin/bash

function abort(){
    echo "An error occurred. Exiting..." 1>&2
    exit 1
}

trap 'abort' 0
set -e
yum -y update
rpm -Uvh http://download.fedoraproject.org/pub/epel/7/x86_64/e/epel-release-7-8.noarch.rpm
yum -y install git gcc gcc-c++ make cmake numpy python-devel openblas-devel protobuf-devel protobuf-python m4 python-pip swig
yum -y install glog-devel gflags-devel
pip install --upgrade pip
pip install wheel

if [ ${WITH_GPU} == 'ON' ]; then
  ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/libcudnn.so
fi

cd ~
git clone https://github.com/baidu/Paddle.git paddle
cd paddle
mkdir build
cd build
cmake .. -DWITH_DOC=OFF -DWITH_GPU=${WITH_GPU} -DWITH_SWIG_PY=ON\
   -DCUDNN_ROOT=/usr/ -DWITH_STYLE_CHECK=OFF -DWITH_AVX=${WITH_AVX}
make -j `nproc`
# because durning make install, there are several warning, so set +e, do not cause abort
make install
echo 'export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}' >> /etc/profile
pip ${PIP_GENERAL_ARGS} install ${PIP_INSTALL_ARGS} /usr/local/opt/paddle/share/wheels/*.whl
paddle version  # print version after build

if [ ${WITH_DEMO} == "ON" ]; then
  pip ${PIP_GENERAL_ARGS} install ${PIP_INSTALL_ARGS}  BeautifulSoup docopt \
    PyYAML pillow
fi
if [ ${IS_DEVEL} == "OFF" ]; then  # clean build packages.
  cd ~
  rm -rf paddle
fi
yum clean packages
trap : 0
