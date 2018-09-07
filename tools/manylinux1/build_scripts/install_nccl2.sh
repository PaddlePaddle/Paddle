#!/bin/bash
VERSION=$(nvcc --version | grep release | grep -oEi "release ([0-9]+)\.([0-9])"| sed "s/release //")
if [ "$VERSION" == "9.0" ]; then
  DEB="nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb"
  URL="http://nccl2-deb.gz.bcebos.com/nccl-repo-ubuntu1604-2.1.15-ga-cuda9.0_1-1_amd64.deb"
else
  DEB="nccl-repo-ubuntu1604-2.1.15-ga-cuda8.0_1-1_amd64.deb"
  URL="http://nccl2-deb.gz.bcebos.com/nccl-repo-ubuntu1604-2.1.15-ga-cuda8.0_1-1_amd64.deb"
fi

DIR="/nccl2"
mkdir -p $DIR
# we cached the nccl2 deb package in BOS, so we can download it with wget
# install nccl2: http://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#down
wget -O $DIR/$DEB $URL

cd $DIR && ar x $DEB && tar xf data.tar.xz
DEBS=$(find ./var/ -name "*.deb")
for sub_deb in $DEBS; do
  echo $sub_deb
  ar x $sub_deb && tar xf data.tar.xz
done
mv -f usr/include/nccl.h /usr/local/include/
mv -f usr/lib/libnccl* /usr/local/lib/
rm -rf $DIR
