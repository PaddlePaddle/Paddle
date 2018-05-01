#!/bin/bash

BUILD_PATH=/paddle/fp16_build_91_7
WHEEL_PATH=$BUILD_PATH/python/dist
INFER_PATH=$BUILD_PATH/paddle/fluid/inference/tests/book
DEMO_PATH=/paddle/python/paddle/fluid/tests/demo/float16

export CUDA_VISIBLE_DEVICES=0

# Build the PaddlePaddle Fluid wheel package and install it.
#mkdir -p $BUILD_PATH && cd $BUILD_PATH
#cmake .. -DWITH_AVX=OFF \
#         -DWITH_MKL=OFF \
#         -DWITH_GPU=ON \
#         -DWITH_TESTING=ON \
#         -DWITH_TIMER=ON \
#         -DWITH_PROFILER=ON \
#         -DWITH_FLUID_ONLY=ON
#make -j `nproc`
#pip install -U "$WHEEL_PATH/$(ls $WHEEL_PATH)"

# Run the demo testing code
cd $DEMO_PATH
stdbuf -oL python float16_inference_accuracy.py \
       --threshold=0.5 \
       --repeat=1 \
       2>&1 | tee -a float16_inference_accuracy.log
#python float16_inference_benchmark.py

exit

# Test inference benchmark of vgg16 on imagenet
$INFER_PATH/test_inference_image_classification_vgg \
    --data_set=imagenet \
    --dirname=$DEMO_PATH/image_classification_imagenet_vgg.inference.model \
    --fp16_dirname=$DEMO_PATH/float16_image_classification_imagenet_vgg.inference.model \
    --repeat=1000 \
    --batch_size=1 \
    --skip_cpu=true \
    2>&1 | tee -a imagenet_vgg_benchmark.log
