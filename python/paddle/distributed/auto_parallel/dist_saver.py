# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import re
import os
import errno
import pickle
import warnings
import logging
import numpy as np
import paddle

from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import static_only
from .utils import get_dist_attr
from .converter import Converter
from .process_group import _g_process_group_map
from ..utils import get_logger


def check_filename(re_exp, filename):
    if re.search(re_exp, filename):
        return True
    else:
        return False


def _process_path(path):
    filename = os.path.basename(path)
    if filename == "":
        raise ValueError(
            "path should be of 'dirname/filename' format, but received filename is empty string"
        )
    try:
        dirname = os.path.dirname(path)
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return dirname, filename


class DistributedSaver:
    def __init__(self):
        self._logger = get_logger(logging.INFO)

    def save(self, path, serial_program, dist_main_program, dist_context):

        dirname, filename = _process_path(path)

        rank_id = paddle.distributed.get_rank()
        # save serial program when rank id is 0
        if rank_id == 0:
            self._save_rank_mapping(dirname)
            serial_model_filename = filename + "_serial.pdmodel"
            serial_model_path = os.path.join(dirname, serial_model_filename)
            with open(serial_model_path, "wb") as f:
                f.write(serial_program.desc.serialize_to_string())

        # save distributed main program
        dist_model_filename = filename + "_dist" + str(rank_id) + ".pdmodel"
        dist_model_path = os.path.join(dirname, dist_model_filename)
        with open(dist_model_path, "wb") as f:
            f.write(dist_main_program.desc.serialize_to_string())

        # save distributed params
        dist_param_filename = filename + "_dist" + str(rank_id) + ".pdparams"
        dist_param_path = os.path.join(dirname, dist_param_filename)
        dist_param = {
            k: np.array(v)
            for k, v in dist_main_program.state_dict().items()
        }
        with open(dist_param_path, "wb") as f:
            pickle.dump(dist_param, f)

        # save distributed attribute
        dist_attr_filename = filename + "_dist" + str(rank_id) + ".pdattr"
        dist_attr_path = os.path.join(dirname, dist_attr_filename)
        dist_attrs = get_dist_attr(dist_main_program, dist_context)
        with open(dist_attr_path, "wb") as f:
            pickle.dump(dist_attrs, f)

        # TODO:save cluster.json

    def load(self,
             path,
             program,
             dist_context,
             strict=True,
             load_optimizer=True):
        # TODO: if `program` is None, load `path.pdmodel`.
        filename = os.path.basename(path)
        if filename == "":
            raise ValueError(
                "path should be of 'dirname/filename' format, but received filename is empty string"
            )
        dirname = os.path.dirname(path)
        # load path.pdparam
        param_file_list = []
        for param_file in os.listdir(dirname):
            if check_filename('{}(.*)_dist(.*).pdparams'.format(filename),
                              param_file):
                param_file_list.append(os.path.join(dirname, param_file))
        param_file_list.sort()
        self._logger.info("Load distributed attribute file: {}".format(
            param_file_list))
        param_dict = {}
        for param_file in param_file_list:
            with open(param_file, 'rb') as f:
                state_dict_info = pickle.load(f, encoding='latin1')
            for name, value in state_dict_info.items():
                if name in param_dict:
                    param_dict[name].append(np.array(value))
                else:
                    param_dict[name] = [np.array(value)]

        # load path.pdattr
        dist_attr_file_list = []
        for dist_attr_file in os.listdir(dirname):
            if check_filename('{}(.*)_dist(.*).pdattr'.format(filename),
                              dist_attr_file):
                dist_attr_file_list.append(
                    os.path.join(dirname, dist_attr_file))
        dist_attr_file_list.sort()
        self._logger.info("Load distributed attribute file: {}".format(
            dist_attr_file_list))
        pre_dist_attr = {}
        for dist_attr_file in dist_attr_file_list:
            with open(dist_attr_file, 'rb') as f:
                dist_attr = pickle.load(f, encoding='latin1')
            for name, attr in dist_attr.items():
                if name not in pre_dist_attr:
                    pre_dist_attr[name] = attr

        # get current dist_attr
        cur_dist_attr = get_dist_attr(program, dist_context)

        # param convert
        converter = Converter(param_dict, pre_dist_attr, cur_dist_attr)
        param_dict = converter.convert(strict=strict)
        program.set_state_dict(param_dict)

    def save_inference_model(self, path, feed_vars, fetch_vars, exe, **kwargs):

        dirname, filename = _process_path(path)

        # save distributed inference program
        rank_id = paddle.distributed.get_rank()
        if rank_id == 0:
            self._save_rank_mapping(dirname)
        op_role_key = core.op_proto_and_checker_maker.kOpRoleAttrName()
        op_role_forward = int(core.op_proto_and_checker_maker.OpRole.Forward)

        dist_main_prog = kwargs.get('program', None)
        if not dist_main_prog:
            dist_main_prog = fluid.default_main_program()
        global_block = dist_main_prog.global_block()

        ops = global_block.ops
        feed_vars_names = list(map(lambda x: x.name, feed_vars))
        fetch_vars_names = list(map(lambda x: x.name, fetch_vars))

        last_idx = -1
        for idx, op in enumerate(ops):
            if op.attr(op_role_key) != op_role_forward:
                continue
            if op.type == "read" or op.type == "feed" or op.type == 'recv_v2':
                feed_vars_names += op.output("Out")
            if op.type == "send_v2":
                fetch_vars_names += op.input("X")
                last_idx = max(idx, last_idx)
            for out_name in op.output_arg_names:
                if out_name in fetch_vars_names:
                    last_idx = max(idx, last_idx)

        used_inputs = []
        used_outputs = []
        for idx, op in enumerate(ops):
            if idx > last_idx:
                break
            used_inputs += op.input_arg_names
            used_outputs += op.output_arg_names

        dist_feed_vars_names = list(set(feed_vars_names) & set(used_inputs))
        dist_fetch_vars_names = list(set(fetch_vars_names) & set(used_outputs))

        dist_feed_vars = [
            global_block.vars[name] for name in dist_feed_vars_names
        ]
        dist_fetch_vars = [
            global_block.vars[name] for name in dist_fetch_vars_names
        ]

        # NOTE: `paddle.static.save_inference_model` does not support subblock.
        dist_filename = filename + "_dist" + str(rank_id)
        dist_path = os.path.join(dirname, dist_filename)
        paddle.static.save_inference_model(
            dist_path,
            dist_feed_vars,
            dist_fetch_vars,
            exe,
            program=dist_main_prog)

    def _save_rank_mapping(self, dirname):
        path = os.path.join(dirname, 'rank_mapping.csv')
        f = open(path, 'w')
        f.write('[ring_id -> ranks]\n')
        for process_group in _g_process_group_map.values():
            ring_id = process_group._group_id
            ranks = [str(rank) for rank in process_group._ranks]
            id_to_rank = str(ring_id) + "," + ",".join(ranks) + '\n'
            f.write(id_to_rank)
            id_to_rank = ""
        f.write('[rank -> ring_ids]\n')
        rank_to_id_dict = {}
        for process_group in _g_process_group_map.values():
            ring_id = process_group._group_id
            for rank in process_group._ranks:
                if rank in rank_to_id_dict:
                    rank_to_id_dict[rank].append(str(ring_id))
                else:
                    rank_to_id_dict[rank] = [str(ring_id)]
        rank_to_id = ""
        for item, val in rank_to_id_dict.items():
            rank_to_id += str(item) + ","
            rank_to_id += ",".join(val) + "\n"
            f.write(rank_to_id)
            rank_to_id = ""
        f.close()
