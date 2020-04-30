export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch train.py
