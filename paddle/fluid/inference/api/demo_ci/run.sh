set -x
PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas
TEST_GPU_CPU=$3 # test both GPU/CPU mode or only CPU mode
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

# download vis_demo data
function download() {
  dir_name=$1
  mkdir -p $dir_name
  cd $dir_name
  wget -q ${URL_ROOT}$dir_name.tar.gz
  tar xzf *.tar.gz
  cd ..
}
URL_ROOT=http://paddlemodels.bj.bcebos.com/inference-vis-demos%2F
mkdir -p data
cd data
vis_demo_list='se_resnext50 ocr mobilenet'
for vis_demo_name in $vis_demo_list; do
  download $vis_demo_name
done
cd ..

# compile and test the demo
mkdir -p build
cd build

for WITH_STATIC_LIB in ON OFF; do
  # -----simple_on_word2vec-----
  rm -rf *
  cmake .. -DPADDLE_LIB=${PADDLE_ROOT}/build/fluid_install_dir/ \
    -DWITH_MKL=$TURN_ON_MKL \
    -DDEMO_NAME=simple_on_word2vec \
    -DWITH_GPU=$TEST_GPU_CPU \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make -j
  word2vec_model=${PADDLE_ROOT}'/build/python/paddle/fluid/tests/book/word2vec.inference.model'
  if [ -d $word2vec_model ]; then
    for use_gpu in $use_gpu_list; do
      ./simple_on_word2vec \
        --dirname=$word2vec_model \
        --use_gpu=$use_gpu
      if [ $? -ne 0 ]; then
        echo "simple_on_word2vec demo runs fail."
        exit 1
      fi
    done
  fi
  # ---------vis_demo---------
  rm -rf *
  cmake .. -DPADDLE_LIB=${PADDLE_ROOT}/build/fluid_install_dir/ \
    -DWITH_MKL=$TURN_ON_MKL \
    -DDEMO_NAME=vis_demo \
    -DWITH_GPU=$TEST_GPU_CPU \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make -j
  for use_gpu in $use_gpu_list; do
    for vis_demo_name in $vis_demo_list; do 
      ./vis_demo \
        --modeldir=../data/$vis_demo_name/model \
        --data=../data/$vis_demo_name/data.txt \
        --refer=../data/$vis_demo_name/result.txt \
        --use_gpu=$use_gpu
      if [ $? -ne 0 ]; then
        echo "vis demo $vis_demo_name runs fail."
        exit 1
      fi
    done
  done
done
set +x
