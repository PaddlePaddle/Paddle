
# PaddlePaddle inference and training script
This directory contains model configuration and tool used to run the PaddlePaddle + NGraph for a local training and inference.

# How to build PaddlePaddle framework with NGraph engine
In order to build the PaddlePaddle + NGraph engine and run proper script follow up a few steps:
1. build the PaddlePaddle project
2. download pre-trained model data
3. set env exports for nGraph and OMP
4. go to script/ directory
5. run the inference/training script

Curently supported models:
* ResNet50 (inference and training).

Short description of aforementioned steps:

## 1. Build paddle
Do it as you usually do. In case you never did it, here are instructions:
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_MKLDNN=ON -DWITH_MKL=ON -DWITH_GOLANG=OFF -DWITH_SWIG_PY=ON -DWITH_STYLE_CHECK=OFF -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_PROFILER=OFF -DWITH_NGRAPH=ON
```
## 2. Download pre-trained model:
In order to download model, go to /save_models directory and run download_resnet50.sh script:
```
$ cd save_models/
$ ./download_resnet50.sh
```

## 3. Set env exports for nGraph and OMP
Set the following exports needed for running nGraph:
```
export FLAGS_use_ngraph=true
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=<num_cpu_cores>
```

## 4. Go to /script directory for scripts
In order to run the scripts, go to /script directory.

## 3. How the benchmark script might be run.
If everything built sucessfully, you can run the following command to start the benchmark job locally:

Run the training job using the nGraph:
```
numactl -l python train_resnet.py \
            --skip_batch_num=<num> \
            --device=CPU \
            --iterations=<num> \
            --pass_num=1 \
            --batch_size=<num> \
            --model=resnet_imagenet \
            --data_set=flowers \
            --use_fake_data \
            --save_model \
            --save_model_path=<path_to_directory_to_save_model>
```
Run the inference job using the nGraph:
```
numactl -l --physcpubind=<num_cpu_cores; from-to> python infer_image_classification.py \
                  --device=CPU \
                  --skip_batch_num=<num> \
                  --iterations=<num> \
                  --batch_size=<batch_size> \
                  --data_set=imagenet \
                  --infer_model_path=<path_to_directory_with_the_model> \
                  --use_fake_data
```
