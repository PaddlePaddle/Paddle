set -x
DEMO_NAME=$1
PADDLE_ROOT=$2
TURN_ON_MKL=$3 # use MKL or Openblas
TEST_GPU_CPU=$4 # test both GPU/CPU mode or only CPU mode
WITH_STATIC_LIB=$5
TENSORRT_INCLUDE_DIR=$6 # TensorRT header file dir, defalut to /usr/local/TensorRT/include
TENSORRT_LIB_DIR=$7 # TensorRT lib file dir, default to /usr/local/TensorRT/lib
inference_install_dir=${PADDLE_ROOT}/build/fluid_install_dir

if [ $TURN_ON_MKL == ON ]; then
  # You can export yourself if move the install path
  MKL_LIB=${inference_install_dir}/third_party/install/mklml/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKL_LIB}
fi
if [ $TEST_GPU_CPU == ON ]; then
  use_gpu_list='true false'
else
  use_gpu_list='false'
fi

USE_TENSORRT=OFF
if [ -d "$TENSORRT_INCLUDE_DIR" -a -d "$TENSORRT_LIB_DIR" ]; then
  USE_TENSORRT=ON
fi

start_dir=${PWD}

if [ ${PWD##*/} == build ]; then
  rm -rf *
else
  mkdir -p build
  cd build
  rm -rf *
fi

echo "pwd: $(pwd)"
# assume we are in a build directory
if [ $DEMO_NAME == simple_on_word2vec ]; then
  # -----simple_on_word2vec-----
  cmake .. -DPADDLE_LIB=${inference_install_dir} \
    -DWITH_MKL=$TURN_ON_MKL \
    -DDEMO_NAME=simple_on_word2vec \
    -DWITH_GPU=$TEST_GPU_CPU \
    -DDEMO_ADD_SRC="" \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make -j
elif [ $DEMO_NAME == vis_demo ]; then
  # ---------vis_demo---------
  cmake .. -DPADDLE_LIB=${inference_install_dir} \
    -DWITH_MKL=$TURN_ON_MKL \
    -DDEMO_NAME=vis_demo \
    -DWITH_GPU=$TEST_GPU_CPU \
    -DDEMO_ADD_SRC="" \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make -j
elif [ $DEMO_NAME == trt_mobilenet_demo ]; then
  # --------tensorrt mobilenet------
  if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
    cmake .. -DPADDLE_LIB=${inference_install_dir} \
      -DWITH_MKL=$TURN_ON_MKL \
      -DDEMO_NAME=trt_mobilenet_demo \
      -DWITH_GPU=$TEST_GPU_CPU \
      -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DUSE_TENSORRT=$USE_TENSORRT \
      -DTENSORRT_INCLUDE_DIR=$TENSORRT_INCLUDE_DIR \
      -DTENSORRT_LIB_DIR=$TENSORRT_LIB_DIR
    make -j
  fi
elif [ $DEMO_NAME == infer_image_classification ]; then
  # -------- infer_image_classification (iic) ------
  cmake .. -DPADDLE_LIB=${inference_install_dir} \
    -DWITH_MKL=$TURN_ON_MKL \
    -DDEMO_NAME=infer_image_classification \
    -DDEMO_ADD_SRC="utils/image_reader.cc;utils/stats.cc" \
    -DWITH_GPU=$TEST_GPU_CPU \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make -j
else
  echo "Unknown demo name ($DEMO_NAME)."
  exit 1
fi

cd ${start_dir}

set +x
