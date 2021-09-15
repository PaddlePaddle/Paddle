rm -rf dp_log/
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch --gpus=0,1 --log_dir=dp_log test_ema.py