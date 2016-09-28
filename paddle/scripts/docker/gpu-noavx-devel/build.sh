#!/bin/bash

function abort(){
    echo "An error occurred. Exiting..." 1>&2
    exit 1
}

trap 'abort' 0
set -e
if [ ${USE_UBUNTU_MIRROR} == "ON" ]; then
    sed -i 's#http://archive\.ubuntu\.com/ubuntu/#mirror://mirrors\.ubuntu\.com/mirrors\.txt#g'\
      /etc/apt/sources.list
fi
apt-get update
apt-get install -y cmake libprotobuf-dev protobuf-compiler git \
    libgoogle-glog-dev libgflags-dev libatlas-dev libatlas3-base g++ m4 python-pip\
    python-protobuf python-numpy python-dev swig

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
  apt-get install -y wget unzip perl python-matplotlib tar xz-utils bzip2 gzip coreutils\
	          sed grep graphviz libjpeg-dev zlib1g-dev
  pip ${PIP_GENERAL_ARGS} install ${PIP_INSTALL_ARGS}  BeautifulSoup docopt \
    PyYAML pillow
fi
if [ ${IS_DEVEL} == "OFF" ]; then  # clean build packages.
  cd ~
  rm -rf paddle
fi
apt-get clean -y
trap : 0
