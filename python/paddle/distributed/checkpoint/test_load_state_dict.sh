rm -rf log/*
rm -f snooper*
python -u -m paddle.distributed.launch --log_dir ./log --devices 1,2,3,4 load_state_dict.py
