#!/bin/bash

BUILD_PATH=/paddle/fp16_build
WHEEL_PATH=$BUILD_PATH/python/dist
INFER_PATH=$BUILD_PATH/paddle/fluid/inference/tests/book
DEMO_PATH=/paddle/paddle/contrib/float16

# Use the single most powerful CUDA GPU on your machine
export CUDA_VISIBLE_DEVICES=0

# Build the PaddlePaddle Fluid wheel package and install it.
mkdir -p $BUILD_PATH && cd $BUILD_PATH
cmake .. -DWITH_AVX=OFF \
         -DWITH_MKL=OFF \
         -DWITH_GPU=ON \
         -DWITH_TESTING=ON \
         -DWITH_PROFILER=ON \
make -j `nproc`
pip install -U "$WHEEL_PATH/$(ls $WHEEL_PATH)"

cd $DEMO_PATH
# Clear previous log results
rm -f *.log

# Test the float16 inference accuracy of resnet32 on cifar10 data set
stdbuf -oL python float16_inference_demo.py \
       --data_set=cifar10 \
       --model=resnet \
       --threshold=0.6 \
       --repeat=10 \
       2>&1 | tee -a float16_inference_accuracy.log

# Sleep to cool down the GPU for consistent benchmarking
sleep 2m

# benchmarking parameters
REPEAT=1000
MAXIMUM_BATCH_SIZE=512

for ((batch_size = 1; batch_size <= MAXIMUM_BATCH_SIZE; batch_size *= 2)); 
do

  # Test inference benchmark of vgg16 on imagenet
  stdbuf -oL python float16_inference_demo.py \
         --data_set=imagenet \
         --model=vgg \
         --threshold=0.001 \
         --repeat=1 \

  $INFER_PATH/test_inference_image_classification_vgg \
      --dirname=$DEMO_PATH/image_classification_imagenet_vgg.inference.model \
      --fp16_dirname=$DEMO_PATH/float16_image_classification_imagenet_vgg.inference.model \
      --repeat=$REPEAT \
      --batch_size=$batch_size \
      --skip_cpu=true \
      2>&1 | tee -a imagenet_vgg16_benchmark.log

  sleep 2m

  # Test inference benchmark of resnet50 on imagenet
  stdbuf -oL python float16_inference_demo.py \
         --data_set=imagenet \
         --model=resnet \
         --threshold=0.001 \
         --repeat=1 \

  $INFER_PATH/test_inference_image_classification_resnet \
      --dirname=$DEMO_PATH/image_classification_imagenet_resnet.inference.model \
      --fp16_dirname=$DEMO_PATH/float16_image_classification_imagenet_resnet.inference.model \
      --repeat=$REPEAT \
      --batch_size=$batch_size \
      --skip_cpu=true \
      2>&1 | tee -a imagenet_resnet50_benchmark.log

  sleep 2m

  # Test inference benchmark of vgg16 on cifar10
  stdbuf -oL python float16_inference_demo.py \
         --data_set=cifar10 \
         --model=vgg \
         --threshold=0.001 \
         --repeat=1 \

  $INFER_PATH/test_inference_image_classification_vgg \
      --dirname=$DEMO_PATH/image_classification_cifar10_vgg.inference.model \
      --fp16_dirname=$DEMO_PATH/float16_image_classification_cifar10_vgg.inference.model \
      --repeat=$REPEAT \
      --batch_size=$batch_size \
      --skip_cpu=true \
      2>&1 | tee -a cifar10_vgg16_benchmark.log

  sleep 1m

  # Test inference benchmark of resnet32 on cifar10
  stdbuf -oL python float16_inference_demo.py \
         --data_set=cifar10 \
         --model=resnet \
         --threshold=0.001 \
         --repeat=1 \

  $INFER_PATH/test_inference_image_classification_vgg \
      --dirname=$DEMO_PATH/image_classification_cifar10_resnet.inference.model \
      --fp16_dirname=$DEMO_PATH/float16_image_classification_cifar10_resnet.inference.model \
      --repeat=$REPEAT \
      --batch_size=$batch_size \
      --skip_cpu=true \
      2>&1 | tee -a cifar10_resnet32_benchmark.log

  sleep 1m

done
