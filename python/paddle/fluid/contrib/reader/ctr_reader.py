#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os

from paddle.fluid import core
from paddle.fluid.executor import global_scope
from paddle.fluid.framework import default_main_program, \
    default_startup_program, Variable
from paddle.fluid.unique_name import generate as unique_name

__all__ = ['ctr_reader']


def monkey_patch_reader_methods(reader):
    def __get_reader__():
        scope = global_scope()
        var = scope.find_var(reader.name)
        return var.get_reader()

    def reset():
        return __get_reader__().reset()

    def start():
        return __get_reader__().start()

    reader.reset = reset
    reader.start = start
    reader.stop_gradient = True
    reader.persistable = True
    return reader


def _copy_reader_var_(block, var):
    new_var = block.create_var(name=var.name, type=core.VarDesc.VarType.READER)
    new_var.desc.set_shapes(var.desc.shapes())
    new_var.desc.set_dtypes(var.desc.dtypes())
    new_var.persistable = True
    return new_var


def ctr_reader(
        feed_dict,
        file_type,  # gzip or plain
        file_format,  # csv or svm
        dense_slot_index,
        sparse_slot_index,
        capacity,
        thread_num,
        batch_size,
        file_list,
        slots,
        name=None):
    """
    Create a CTR reader for data feeding in Python

    This layer returns a Reader Variable.
    The Reader provides :code:`decorate_paddle_reader()` and
    :code:`decorate_tensor_provider()` to set a Python generator as the data
    source in Python side. When :code:`Executor::Run()` is invoked in C++
    side, the data from the generator would be read automatically. Unlike
    :code:`DataFeeder.feed()`, the data reading process and
    :code:`Executor::Run()` process can run in parallel using
    :code:`py_reader`. The :code:`start()` method of the Reader should be
    called when each pass begins, while the :code:`reset()` method should be
    called when the pass ends and :code:`fluid.core.EOFException` raises.
    Note that :code:`Program.clone()` method cannot clone :code:`py_reader`.

    Args:
       feed_dict(list(variable)): a list of data variable.
       file_type('gzip'|'plain'): the type of the data file
       file_format('csv'|'svm'): csv data or svm data format.
        cvs data format is :
            label dense_fea,dense_fea sparse_fea,sparse_fea
        the svm data format is :
            label slot1:fea_sign slot2:fea_sign slot1:fea_sign
       dense_slot_index(list(int)): the index of dense slots
       sparse_slot_index(list(int)): the index of sparse slots
       capacity(int): The buffer capacity maintained by :code:`py_reader`.
       thread_num(int): the thread num to read files by cpp reader.
       batch_size(int): batch size of data.
       file_list(list(str)): List of file names that need to read.
       slots(list(int64)): list of slot id.
       name(string): The prefix Python queue name and Reader name. None will
            be generated automatically.

    Returns:
       Variable: A Reader from which we can get feeding data.

    Examples:

        1. The basic usage of :code:`ctr_reader` is as follows:

     .. code-block:: python

        py_reader = fluid.contrib.ctr_reader.ctr_reader(
          feed_dict=datas, file_type='plain', file_format='csv',
          file_list=file_list, dense_slot_indexs=[1, 2, 3, 4], sparse_slot_indexs=[],
          capacity=64, thread_num=20, batch_size=1000, slots=[], name='ctr_reader')

    """
    if name is None:
        queue_name = unique_name('lod_tensor_blocking_queue')
        reader_name = unique_name('create_ctr_reader')
    else:
        queue_name = "_".join([name, "queue"])
        reader_name = "_".join([name, "reader"])

    var = global_scope().var(queue_name)
    feed_queue = core.init_lod_tensor_blocking_queue(
        int(os.getenv("CPU_NUM", "1")), var, capacity)

    startup_blk = default_startup_program().current_block()
    reader_var = startup_blk.create_var(name=reader_name)
    startup_blk.append_op(
        type='create_ctr_reader',
        inputs={'blocking_queue': [queue_name]},
        outputs={'Out': [reader_var]},
        attrs={
            'use_data_config': False,
            'thread_num': thread_num,
            'batch_size': batch_size,
            'file_list': file_list,
            'file_type': file_type,
            'file_format': file_format,
            'dense_slot_index': dense_slot_index,
            'sparse_slot_index': sparse_slot_index,
            'sparse_slots': slots,
            'ranks': [],
            'lod_levels': [],
            'shape_concat': []
        })

    dtypes = [data.dtype for data in feed_dict]
    reader_var.desc.set_dtypes(dtypes)
    reader_var.persistable = True

    main_prog_reader_var = _copy_reader_var_(
        default_main_program().current_block(), reader_var)

    reader = monkey_patch_reader_methods(main_prog_reader_var)

    # monkey patch py_reader special methods
    reader.queue = feed_queue
    reader.exited = False

    main_blk = default_main_program().current_block()
    main_blk.append_op(
        type='read',
        inputs={'Reader': [reader]},
        attrs={'infer_out': False},
        outputs={'Out': feed_dict})

    return reader
