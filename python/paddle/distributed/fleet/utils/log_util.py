# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import subprocess
from logging.handlers import RotatingFileHandler

import paddle
from paddle.distributed.utils.log_utils import get_logger

logger = get_logger("INFO", __name__)


def set_log_level(level):
    """
    Set log level

    Args:
        level (str|int): a specified level

    Example 1:
        import paddle
        import paddle.distributed.fleet as fleet
        fleet.init()
        fleet.setLogLevel("DEBUG")

    Example 2:
        import paddle
        import paddle.distributed.fleet as fleet
        fleet.init()
        fleet.setLogLevel(1)

    """
    assert isinstance(level, (str, int)), "level's type must be str or int"
    if isinstance(level, int):
        logger.setLevel(level)
    else:
        logger.setLevel(level.upper())


def get_log_level_code():
    """
    Return current log level code
    """
    return logger.getEffectiveLevel()


def get_log_level_name():
    """
    Return current log level name
    """
    return logging.getLevelName(get_log_level_code())


def layer_to_str(base, *args, **kwargs):
    name = base + "("
    if args:
        name += ", ".join(str(arg) for arg in args)
        if kwargs:
            name += ", "
    if kwargs:
        name += ", ".join(f"{key}={value}" for key, value in kwargs.items())
    name += ")"
    return name


class DistributedLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def info(self, msg, *args, **kwargs):
        paddle.device.synchronize()
        super().info(f"Distributed Debug: {msg}", *args, **kwargs)


def get_rotate_file_logger(log_level, name='root'):
    distributed_logger = DistributedLogger(name + '_rotate', level=log_level)
    distributed_logger.propagate = False

    device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
    log_dir = os.path.join(os.getcwd(), "hybrid_parallel")
    os.makedirs(log_dir, exist_ok=True)

    path = os.path.join(log_dir, f"worker_{device_id}.log")
    handler = RotatingFileHandler(
        path, maxBytes=2 * 1024 * 1024 * 1024, backupCount=3  # 2GB
    )

    log_format = logging.Formatter(
        '[%(asctime)-15s] [%(levelname)8s] %(filename)s:%(lineno)s - %(message)s'
    )
    handler.setFormatter(log_format)
    distributed_logger.addHandler(handler)
    return distributed_logger


g_sync_rotate_logger = None


def get_sync_logger():
    global logger
    paddle.device.synchronize()
    return logger


def sync_rotate_logger():
    global g_sync_rotate_logger
    if g_sync_rotate_logger is None:
        g_sync_rotate_logger = get_rotate_file_logger("INFO", __name__)
    return g_sync_rotate_logger


def check_memory_usage(msg=""):
    GB = 1024.0 * 1024.0 * 1024.0
    mem_dict = {}
    mem_dict['max_memory_allocated_size'] = (
        paddle.device.cuda.max_memory_allocated() / GB
    )
    mem_dict['max_memory_reserved_size'] = (
        paddle.device.cuda.max_memory_reserved() / GB
    )
    mem_dict['memory_allocated_size'] = (
        paddle.device.cuda.memory_allocated() / GB
    )
    mem_dict['memory_reserved_size'] = paddle.device.cuda.memory_reserved() / GB
    mem_msg = f"checking gpu memory usage {msg}:"
    for key in mem_dict:
        mem_msg += f"\n{key}: {mem_dict[key]}GB"
    logger.info(mem_msg)

    if hasattr(paddle.device.cuda, 'max_pinned_memory_allocated'):
        mem_dict = {}
        mem_dict['max_memory_allocated_size'] = (
            paddle.device.cuda.max_pinned_memory_allocated() / GB
        )
        mem_dict['max_memory_reserved_size'] = (
            paddle.device.cuda.max_pinned_memory_reserved() / GB
        )
        mem_dict['memory_allocated_size'] = (
            paddle.device.cuda.pinned_memory_allocated() / GB
        )
        mem_dict['memory_reserved_size'] = (
            paddle.device.cuda.pinned_memory_reserved() / GB
        )
        mem_msg = f"checking pinned memory usage {msg}:"
        for key in mem_dict:
            mem_msg += f"\n{key}: {mem_dict[key]}GB"
        logger.info(mem_msg)

    if hasattr(paddle.device, 'cpu') and hasattr(
        paddle.device.cpu, 'max_memory_allocated'
    ):
        mem_dict = {}
        mem_dict['max_memory_allocated_size'] = (
            paddle.device.cpu.max_memory_allocated() / GB
        )
        mem_dict['max_memory_reserved_size'] = (
            paddle.device.cpu.max_memory_reserved() / GB
        )
        mem_dict['memory_allocated_size'] = (
            paddle.device.cpu.memory_allocated() / GB
        )
        mem_dict['memory_reserved_size'] = (
            paddle.device.cpu.memory_reserved() / GB
        )
        mem_msg = f"checking cpu memory usage {msg}:"
        for key in mem_dict:
            mem_msg += f"\n{key}: {mem_dict[key]}GB"
        logger.info(mem_msg)

    # Execute the command and get the output
    result = subprocess.run(["free", "-h"], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')

    # Extract data
    mem_data = lines[1].split()
    swap_data = lines[2].split()

    # Format and print
    formatted_output = f"checking CPU memory usage: {msg} Memory - Total: {mem_data[1]}, Used: {mem_data[2]}, Free: {mem_data[3]} Available:{mem_data[-1]}"
    logger.info(formatted_output)
