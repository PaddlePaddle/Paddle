import os

__all__ = ['DATA_HOME']

DATA_HOME = os.path.expanduser('~/.cache/paddle_data_set')

if not os.path.exists(DATA_HOME):
    os.makedirs(DATA_HOME)
