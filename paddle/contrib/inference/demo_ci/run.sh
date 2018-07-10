set -x
PADDLE_ROOT=$1
WITH_MKL=$2
WITH_GPU=$3
if [ $3 == "ON" ]; then
  use_gpu_list='true false'
else    
  use_gpu_list='false'
fi

mkdir -p build
cd build

for WITH_STATIC_LIB in true false; do
  rm -rf *
  cmake .. -DPADDLE_LIB=${PADDLE_ROOT}/build/fluid_install_dir/ \
    -DWITH_MKL=$WITH_MKL \
    -DDEMO_NAME=simple_on_word2vec \
    -DWITH_GPU=$WITH_GPU \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make
  for use_gpu in $use_gpu_list; do
    ./simple_on_word2vec \
      --dirname=${PADDLE_ROOT}/build/python/paddle/fluid/tests/book/word2vec.inference.model \
      --use_gpu=$use_gpu
  done
done
if [ $? -eq 0 ]; then
  exit 0
else
  echo "inference demo runs fail."
  exit 1
fi
set +x
