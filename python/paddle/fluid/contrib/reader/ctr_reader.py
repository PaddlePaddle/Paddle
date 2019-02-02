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

from paddle.fluid import core
from paddle.fluid.executor import global_scope
from paddle.fluid.framework import default_main_program, \
    default_startup_program, Variable
from paddle.fluid.unique_name import generate as unique_name


def monkey_patch_reader_methods(reader):
    def __get_reader__():
        scope = global_scope()
        var = scope.find_var(reader.name)
        return var.get_reader()

    def reset():
        return __get_reader__().reset()

    reader.reset = reset
    reader.stop_gradient = True
    reader.persistable = True
    return reader


def _copy_reader_var_(block, var):
    new_var = block.create_var(name=var.name, type=core.VarDesc.VarType.READER)
    new_var.desc.set_shapes(var.desc.shapes())
    new_var.desc.set_dtypes(var.desc.dtypes())
    new_var.persistable = True
    return new_var


def ctr_reader(feed_data,
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
       capacity(int): The buffer capacity maintained by :code:`py_reader`.
       thread_num(list|tuple): List of tuples which declaring data shapes.
       batch_size(list|tuple): List of strs which declaring data type.
       file_list(list|tuple): List of ints which declaring data lod_level.
       slots(bool): Whether use double buffer or not.
       name(basestring): The prefix Python queue name and Reader name. None will
            be generated automatically.

    Returns:
       Variable: A Reader from which we can get feeding data.

    Examples:

        1. The basic usage of :code:`py_reader` is as follows:
    """
    if name is None:
        queue_name = unique_name('lod_tensor_blocking_queue')
        reader_name = unique_name('create_ctr_reader')
    else:
        queue_name = "_".join([name, "queue"])
        reader_name = "_".join([name, "reader"])

    var = global_scope().var(queue_name)
    feed_queue = core.init_lod_tensor_blocking_queue(var, capacity, shapes)

    startup_blk = default_startup_program().current_block()
    reader_var = startup_blk.create_var(name=reader_name)
    startup_blk.append_op(
        type='create_ctr_reader',
        inputs={'blocking_queue': [queue_name]},
        outputs={'Out': [reader_var]},
        attrs={
            'thread_num': thread_num,
            'batch_size': batch_size,
            'file_list': file_list,
            'slots': slots,
        })

    reader_var.persistable = True

    main_prog_reader_var = _copy_reader_var_(
        default_main_program().current_block(), reader_var)

    reader = monkey_patch_reader_methods(main_prog_reader_var)

    # monkey patch py_reader special methods
    reader.queue = feed_queue
    reader.exited = False

    main_blk = default_main_program().current_block()
    main_blk.append_op(
        type='read', inputs={'Reader': [reader]}, outputs={'Out': feed_data})

    return reader
