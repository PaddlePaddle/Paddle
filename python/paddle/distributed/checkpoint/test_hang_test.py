rm -rf log/*
rm -f snooper*
python -u -m paddle.distributed.launch --log_dir ./log --devices 0,1,2,3 test.py
