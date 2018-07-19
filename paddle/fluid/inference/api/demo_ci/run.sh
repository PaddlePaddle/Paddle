set -x
PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas
TEST_GPU_CPU=$3 # test both GPU/CPU mode or only CPU mode
use_mkldnn='false'
if [ $2 == ON ]; then
  # You can export yourself if move the install path
  MKL_LIB=${PADDLE_ROOT}/build/fluid_install_dir/third_party/install/mklml/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKL_LIB}
  MKLDNN_PATH=${PADDLE_ROOT}/build/fluid_install_dir/third_party/install/mkldnn
  if [ -d $MKLDNN_PATH ]; then
    use_mkldnn='true'
  fi
fi
if [ $3 == ON ]; then
  use_gpu_list='true false'
else    
  use_gpu_list='false'
fi

# cmake .. & make
function compile_demo() {
  demo_name=$1
  WITH_STATIC_LIB=$2
  rm -rf *
  cmake .. -DPADDLE_LIB=${PADDLE_ROOT}/docker_build/fluid_install_dir/ \
    -DWITH_MKL=$TURN_ON_MKL \
    -DDEMO_NAME=$demo_name \
    -DWITH_GPU=$TEST_GPU_CPU \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make -j
}

# test demo in CPU, GPU and MKLDNN environment
function test_demo() {
  demo_name=$1
  model_dir=$2
  test_mkldnn=$3
  function report() {
    log=$1
    if [ $log -ne 0 ]; then
      echo "$demo_name runs fail on $model_dir."
      exit 1
    else
      echo "$demo_name runs success on $model_dir."
    fi
  }
  for use_gpu in $use_gpu_list; do
    ./$demo_name --dirname=$model_dir --use_gpu=$use_gpu
    report $?
  done
  if [ $use_mkldnn == true ] && [ $test_mkldnn == true ]; then
    ./$demo_name --dirname=$model_dir --use_gpu=false --use_mkldnn=true
    report $?
  fi
}

# download vis_demo data
function download() {
  dir_name=$1
  URL_ROOT=http://paddlemodels.bj.bcebos.com/inference-vis-demos%2F
  mkdir -p data
  cd data
  mkdir -p $dir_name
  cd $dir_name
  wget -q ${URL_ROOT}$dir_name.tar.gz
  tar xzf *.tar.gz
  cd ../..
}
vis_demo_list='se_resnext50 ocr mobilenet'
#for vis_demo_name in $vis_demo_list; do
#  download $vis_demo_name
#done

# compile and test the demo
mkdir -p build
cd build

for WITH_STATIC_LIB in OFF; do
  # -----simple_on_word2vec-----
  compile_demo simple_on_word2vec $WITH_STATIC_LIB
  word2vec_model=${PADDLE_ROOT}'/build/python/paddle/fluid/tests/book/word2vec.inference.model'
  if [ -d $word2vec_model ]; then
    test_demo simple_on_word2vec $word2vec_model true 
  fi
  # ---------vis_demo---------
  compile_demo vis_demo $WITH_STATIC_LIB
  # only mobilenet support MKLDNN inference now.
  test_demo vis_demo ../data/mobilenet true
  # group convolution is not implemented yet at conv_mkldnn_op.cc:247
  test_demo vis_demo ../data/se_resnext50 false
  # TODO(luotao): terminate called after throwing an instance of 'mkldnn::error'
  test_demo vis_demo ../data/ocr false
done
set +x
