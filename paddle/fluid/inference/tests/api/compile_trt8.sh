#export LD_LIBRARY_PATH=/root/lvmengsi/envs/trt8/TensorRT-8.2.0.6/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/lvmengsi/envs/trt8/trt8.0_compile/TensorRT-8.0_Opensource/build_false/trt8_compile_all_output/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/root/lvmengsi/envs/trt8/trt8.0_compile/TensorRT-8.0_Opensource/build_true/trt8_compile_all_output/lib/:$LD_LIBRARY_PATH
rm -rf build

mkdir -p build
cd build
rm -rf *

#DEMO_NAME=ernie_xnli_new_pred
#DEMO_NAME=ernie_xnli_new_pred_qps
DEMO_NAME=trt_dynamic_shape_ernie_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

#LIB_DIR=/root/lvmengsi/Paddle_lms/Paddle_dev/Paddle/build_trt8/paddle_inference_install_dir/
#LIB_DIR=/root/lvmengsi/Paddle_lms/Paddle_0609/Paddle/build_trt8/paddle_inference_install_dir/
LIB_DIR=/root/zhanghandi/envs/Paddle/build/paddle_inference_install_dir/
CUDNN_LIB=/root/lvmengsi/envs/trt8/cudnn-8.2/lib64/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/root/lvmengsi/envs/trt8/TensorRT-8.2.0.6/
#TENSORRT_ROOT=/root/lvmengsi/envs/trt8/trt8.0_compile/TensorRT-8.0_Opensource/build_false/trt8_compile_all_output/

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j
