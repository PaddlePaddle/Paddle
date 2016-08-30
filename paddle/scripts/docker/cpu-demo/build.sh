#!/bin/bash

function abort(){
    echo "An error occurred. Exiting..." 1>&2
    exit 1
}

trap 'abort' 0
set -e
sed -i 's#http://archive\.ubuntu\.com/ubuntu/#mirror://mirrors\.ubuntu\.com/mirrors\.txt#g' /etc/apt/sources.list
apt-get update
apt-get install -y cmake libprotobuf-dev protobuf-compiler git \
    libgoogle-glog-dev libgflags-dev libatlas-dev libatlas3-base g++ m4 python-pip\
    python-protobuf python-numpy python-dev swig

if [ ${WITH_GPU} == "ON" ]; then  # install cuda
  cd ~
  apt-get install -y aria2 wget
  echo "Downloading cuda tookit"
  set +e
  for ((i=0; i<100; i++))
  do
    aria2c -x 10 -s 10 --lowest-speed-limit=${LOWEST_DL_SPEED} http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
    if [ $? -eq 0 ]; then
       break
    fi
  done

  set -e
  wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run.md5
  md5sum -c cuda_7.5.18_linux.run.md5
  chmod +x cuda_7.5.18_linux.run
  ./cuda_7.5.18_linux.run --extract=$PWD
  ./cuda-linux64-rel-7.5.18-19867135.run -noprompt
  rm *.run *.run.md5

  echo "Downloading cudnn v5.1"
  set +e
  for ((i=0; i<100; i++))
  do
    aria2c -x 10 --lowest-speed-limit=${LOWEST_DL_SPEED} http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-7.5-linux-x64-v5.1.tgz
    if [ $? -eq 0 ]; then
        break
    fi
  done
  set -e
  echo "$CUDNN_DOWNLOAD_SUM  cudnn-7.5-linux-x64-v5.1.tgz" | sha256sum -c --strict -
  tar -xzf cudnn-7.5-linux-x64-v5.1.tgz -C /usr/local
  rm cudnn-7.5-linux-x64-v5.1.tgz
  ldconfig
  export PATH=/usr/local/cuda/bin:$PATH
  apt-get purge -y aria2
fi
set -e
cd ~
git clone https://github.com/baidu/Paddle.git paddle
cd paddle
mkdir build
cd build
cmake .. -DWITH_DOC=OFF -DWITH_GPU=${WITH_GPU} -DWITH_SWIG_PY=ON
make -j `nproc`
# because durning make install, there are several warning, so set +e, do not cause abort
make install
echo 'export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}' >> /etc/profile
pip ${PIP_GENERAL_ARGS} install ${PIP_INSTALL_ARGS} /usr/local/opt/paddle/share/wheels/*.whl
paddle version  # print version after build

if [ ${WITH_DEMO} == "ON" ]; then
  apt-get install -y wget unzip perl python-matplotlib tar xz-utils bzip2 gzip coreutils\
	          sed grep graphviz 
  pip ${PIP_GENERAL_ARGS} install ${PIP_INSTALL_ARGS}  BeautifulSoup docopt PyYAML
fi
if [ ${IS_DEVEL} == "OFF" ]; then  # clean build packages.
  cd ~
  # TODO(yuyang18): Do clean for devel package, and cuda devel tools
  rm -rf paddle
fi
apt-get clean -y
trap : 0
