set -x
PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas
TEST_GPU_CPU=$3 # test both GPU/CPU mode or only CPU mode
DATA_DIR=$4 # dataset
TENSORRT_INCLUDE_DIR=$5 # TensorRT header file dir, defalut to /usr/local/TensorRT/include
TENSORRT_LIB_DIR=$6 # TensorRT lib file dir, default to /usr/local/TensorRT/lib
inference_install_dir=${PADDLE_ROOT}/build/fluid_install_dir

cd `dirname $0`
current_dir=`pwd`
if [ $2 == ON ]; then
  # You can export yourself if move the install path
  MKL_LIB=${inference_install_dir}/third_party/install/mklml/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKL_LIB}
fi
if [ $3 == ON ]; then
  use_gpu_list='true false'
else
  use_gpu_list='false'
fi

USE_TENSORRT=OFF
if [ -d "$TENSORRT_INCLUDE_DIR" -a -d "$TENSORRT_LIB_DIR" ]; then
  USE_TENSORRT=ON
fi

PREFIX=inference-vis-demos%2F
URL_ROOT=http://paddlemodels.cdn.bcebos.com/${PREFIX}

# download vis_demo data
function download() {
  dir_name=$1
  mkdir -p $dir_name
  cd $dir_name
  if [[ -e "${PREFIX}${dir_name}.tar.gz" ]]; then
    echo "${PREFIX}{dir_name}.tar.gz has been downloaded."
  else
      wget -q ${URL_ROOT}$dir_name.tar.gz
      tar xzf *.tar.gz
  fi
  cd ..
}

mkdir -p $DATA_DIR
cd $DATA_DIR

vis_demo_list='se_resnext50 ocr mobilenet'
for vis_demo_name in $vis_demo_list; do
  download $vis_demo_name
done

ic_model_list='ResNet50 SE-ResNeXt50 MobileNet-v1'

build_demo() {
  echo "WITH_STATIC_LIB: $WITH_STATIC_LIB"
  . ${current_dir}/build.sh $1 $PADDLE_ROOT $TURN_ON_MKL $TEST_GPU_CPU $WITH_STATIC_LIB $TENSORRT_INCLUDE_DIR $TENSORRT_LIB_DIR
}

# compile and test the demo
cd $current_dir

for WITH_STATIC_LIB in ON OFF; do
# TODO(Superjomn) reopen this
# something wrong with the TensorArray reset.
:<<D
  # -----simple_on_word2vec-----
  build_demo simple_on_word2vec
  word2vec_model=$DATA_DIR'/word2vec/word2vec.inference.model'
  if [ -d $word2vec_model ]; then
    for use_gpu in $use_gpu_list; do
      ./build/simple_on_word2vec \
	--dirname=$word2vec_model \
	--use_gpu=$use_gpu
      if [ $? -ne 0 ]; then
	echo "simple_on_word2vec demo runs fail."
	exit 1
      fi
    done
  fi
D

  # ---------vis_demo---------
  build_demo vis_demo
  for use_gpu in $use_gpu_list; do
    for vis_demo_name in $vis_demo_list; do
      ./build/vis_demo \
	--modeldir=$DATA_DIR/$vis_demo_name/model \
	--data=$DATA_DIR/$vis_demo_name/data.txt \
	--refer=$DATA_DIR/$vis_demo_name/result.txt \
	--use_gpu=$use_gpu
      if [ $? -ne 0 ]; then
	echo "vis demo $vis_demo_name runs fail."
	exit 1
      fi
    done
  done

  # --------tensorrt mobilenet------
  if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
    build_demo trt_mobilenet_demo
    ./build/trt_mobilenet_demo \
      --modeldir=$DATA_DIR/mobilenet/model \
      --data=$DATA_DIR/mobilenet/data.txt \
      --refer=$DATA_DIR/mobilenet/result.txt 
  fi

  #--------- infer_image_classification ---------
  build_demo infer_image_classification
  for ic_model in $ic_model_list; do
    MODEL_DIR=${DATA_DIR}"/"${ic_model}
    IMAGENET_DIR=$DATA_DIR"/ImageNet/"
    if [ -d $MODEL_DIR -a -d $IMAGENET_DIR ]; then
      ./build/infer_image_classification \
        --infer_model=$MODEL_DIR \
        --data_list=$IMAGENET_DIR"/val_list.txt" \
        --data_dir=$IMAGENET_DIR \
        --batch_size=50 \
        --num_threads=14 \
        --skip_batch_num=0 \
        --iterations=10  \
        --profile \
        --with_mkldnn=1 \
        --with_labels=1
    fi
  done
    
done
set +x
