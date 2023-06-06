# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import shutil

import paddle

CheckpointMetaName = "latest_checkpoint.pdmeta"


def _is_complete_checkpoint(file_list, file_prefix="default"):
    rank = paddle.distributed.get_rank()
    if rank == 0 and f"{file_prefix}_serial.pdmodel" not in file_list:
        return False
    if f"{file_prefix}_dist{rank}.pdmodel" not in file_list:
        return False
    if f"{file_prefix}_dist{rank}.pdparams" not in file_list:
        return False
    if f"{file_prefix}_dist{rank}.pdattr" not in file_list:
        return False
    if f"{file_prefix}_dist{rank}.pdopt" not in file_list:
        return False
    return True


def _get_checkpoint_directory(file_dir):
    checkpoint_dir_list = os.listdir(file_dir)
    checkpoint_dir_list = list(
        filter(lambda x: x != CheckpointMetaName, checkpoint_dir_list)
    )
    if len(checkpoint_dir_list) == 0:
        return None
    return checkpoint_dir_list


def _get_checkpoint_prefix(file_dir):
    file_list = os.listdir(file_dir)
    file_prefix_map = {}
    for file_name in file_list:
        if file_name in ["rank_mapping.csv"]:
            continue
        file_name_split = file_name.split("_")
        file_name_prefix = "_".join(file_name_split[:-1])
        if file_name_prefix not in file_prefix_map:
            file_prefix_map[file_name_prefix] = []
        file_prefix_map[file_name_prefix].append(file_name)
    return file_prefix_map


def get_latest_checkpoint_prefix(file_dir, rank_size):
    if not os.path.exists(file_dir):
        return None
    checkpoint_dir_list = _get_checkpoint_directory(file_dir)
    if checkpoint_dir_list is None:
        return None

    checkpoint_dir_list = [
        os.path.join(file_dir, dir) for dir in checkpoint_dir_list
    ]
    checkpoint_dir_list = sorted(
        checkpoint_dir_list, key=os.path.getmtime, reverse=True
    )
    latest_checkpoint_dir_path = checkpoint_dir_list[0]
    checkpoint_prefix_map = _get_checkpoint_prefix(latest_checkpoint_dir_path)
    for prefix, files in checkpoint_prefix_map.items():
        full_path = os.path.join(latest_checkpoint_dir_path, prefix)
        if _is_complete_checkpoint(files, "default"):
            return full_path
        else:
            logging.error(
                f"Get latest checkpoint failed, missing some model files. checkpoint_dir: {full_path}"
            )
    return None


def update_checkpoint_filelist(file_dir, latest_path, keep_checkpoint_max_num):
    checkpoint_dir_list = _get_checkpoint_directory(file_dir)
    if checkpoint_dir_list is None:
        return None

    checkpoint_dir_list = [
        os.path.join(file_dir, dir) for dir in checkpoint_dir_list
    ]
    checkpoint_dir_list = sorted(
        checkpoint_dir_list, key=os.path.getmtime, reverse=True
    )

    if len(checkpoint_dir_list) > keep_checkpoint_max_num:
        for checkpoint_dir in checkpoint_dir_list[keep_checkpoint_max_num:]:
            rmdir = os.path.join(file_dir, checkpoint_dir)
            shutil.rmtree(rmdir)


def get_checkpoint_meta_path(checkpoint_meta_dir):
    return os.path.join(checkpoint_meta_dir, CheckpointMetaName)
