import os

__all__ = ['DATA_HOME']

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset')

if not os.path.exists(DATA_HOME):
    os.makedirs(DATA_HOME)
