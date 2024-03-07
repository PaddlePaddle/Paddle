# export PYTHONPATH=/root/paddlejob/workspace/yinwei/Paddle/test:$PYTHONPATH
python -m paddle.distributed.launch --gpu "0,1" /root/paddlejob/workspace/yinwei/Paddle/test/collective/fleet/test_parallel_dygraph_tensor_parallel.py

