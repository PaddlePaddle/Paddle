rm -rf log/*
python -u -m paddle.distributed.launch --log_dir ./log --devices 0,1 save_state_dict.py
