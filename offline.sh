export PYTHONPATH=/root/paddlejob/workspace/env_run/output/lizexu/Paddle-feisheng/test/legacy_test:$PYTHONPATH
export PYTHONPATH=/root/paddlejob/workspace/env_run/output/lizexu/Paddle-feisheng:$PYTHONPATH
python -m pytest -sv test/legacy_test/test_cuda_graph.py