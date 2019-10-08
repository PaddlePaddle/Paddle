cmake .. -DWITH_FLUID_ONLY=ON \
  -DWITH_MKL=OFF \
  -DWITH_GPU=ON \
  -DWITH_TESTING=ON \
  -DWITH_STYLE_CHECK=OFF \
  -DWITH_FAST_BUNDLE_TEST=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH_NAME=Auto \
  # -DCUDNN_ROOT=/home/work/cudnn/cudnn_v6/cuda \
  # -DTHIRD_PARTY_PATH=/home/users/dengkaipeng/.third_party \
  # -DCMAKE_INSTALL_PREFIX=`pwd`/output \

make -j10
