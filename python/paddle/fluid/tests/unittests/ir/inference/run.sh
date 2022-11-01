#CUDA_PATH=/home/work/paddle_compile_env/cuda-11.1
CUDA_PATH=/home/disk1/zhangxin/myenv/cuda-10.2
CUDNN=ON

#CUDNN_ROOT=/home/work/paddle_compile_env/cudnn_v8.1.1_cuda11.1/cuda
CUDNN_ROOT=/home/disk1/zhangxin/myenv/cudnn_v8.1.1_cuda10.2/cuda

export PYTHONPATH=/home/disk1/zhangxin/paddle_custome_bianyi_roformer/Paddle/build/python
#export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:/home/work/paddle_compile_env/TensorRT-7.2.3.4-cuda11.1/lib:/home/work/paddle_compile_env/cudnn_v8.1.1_cuda11.1/cuda/lib64
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:/home/disk1/zhangxin/myenv/TensorRT-7.2.3.4-cuda10.2/lib:/home/disk1/zhangxin/myenv/cudnn_v8.1.1_cuda10.2/cuda/lib64
export CUDA_VISIBLE_DEVICES=1

python test_multihead_matmul_roformer_fuse_pass.py
