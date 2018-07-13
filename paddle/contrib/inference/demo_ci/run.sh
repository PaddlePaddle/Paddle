set -x
PADDLE_ROOT=$1
WITH_MKL=$2
WITH_GPU=$3
if [ $2 == ON ]; then
  # You can export yourself if move the install path
  MKL_LIB=${PADDLE_ROOT}/build/fluid_install_dir/third_party/install/mklml/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKL_LIB}
fi
if [ $3 == ON ]; then
  use_gpu_list='true false'
else    
  use_gpu_list='false'
fi

mkdir -p build
cd build

for WITH_STATIC_LIB in ON OFF; do
  rm -rf *
  cmake .. -DPADDLE_LIB=${PADDLE_ROOT}/build/fluid_install_dir/ \
    -DWITH_MKL=$WITH_MKL \
    -DDEMO_NAME=simple_on_word2vec \
    -DWITH_GPU=$WITH_GPU \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make -j
  for use_gpu in $use_gpu_list; do
    ./simple_on_word2vec \
      --dirname=${PADDLE_ROOT}/build/python/paddle/fluid/tests/book/word2vec.inference.model \
      --use_gpu=$use_gpu
    if [ $? -ne 0 ]; then
      echo "inference demo runs fail."
      exit 1
    fi
  done
done
set +x
