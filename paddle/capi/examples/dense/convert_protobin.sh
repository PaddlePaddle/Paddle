#!/bin/bash
python -m paddle.utils.dump_config trainer_config.py '' --binary > trainer_config.bin
