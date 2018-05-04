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

from .. import core
from ..framework import convert_np_dtype_to_dtype_, default_main_program, default_startup_program, Program
from ..unique_name import generate as unique_name
from control_flow import BlockGuard
from ..layer_helper import LayerHelper
from ..executor import global_scope

__all__ = [
    'data', 'BlockGuardServ', 'ListenAndServ', 'Send', 'open_recordio_file',
    'open_files', 'read_file', 'shuffle', 'batch', 'double_buffer'
]


def data(name,
         shape,
         append_batch_size=True,
         dtype='float32',
         lod_level=0,
         type=core.VarDesc.VarType.LOD_TENSOR,
         stop_gradient=True):
    """
    **Data Layer**

    This function takes in the input and based on whether data has
    to be returned back as a minibatch, it creates the global variable by using
    the helper functions. The global variables can be accessed by all the
    following operators in the graph.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    Args:
       name(str): The name/alias of the function
       shape(list): Tuple declaring the shape.
       append_batch_size(bool): Whether or not to append the data as a batch.
       dtype(int|float): The type of data : float32, float_16, int etc
       type(VarType): The output type. By default it is LOD_TENSOR.
       lod_level(int): The LoD Level. 0 means the input data is not a sequence.
       main_program(Program): Name of the main program that calls this
       startup_program(Program): Name of the startup program
       stop_gradient(bool): A boolean that mentions whether gradient should flow.

    Returns:
        Variable: The global variable that gives access to the data.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name='x', shape=[784], dtype='float32')
    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in xrange(len(shape)):
        if shape[i] is None:
            shape[i] = -1
            append_batch_size = False
        elif shape[i] < 0:
            append_batch_size = False

    if append_batch_size:
        shape = [-1] + shape  # append batch size as -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=type,
        stop_gradient=stop_gradient,
        lod_level=lod_level)


class BlockGuardServ(BlockGuard):
    """
    BlockGuardServ class.

    BlockGuardServ class is used to create an op with a block in a program.
    """

    def __init__(self, server):
        if not (isinstance(server, ListenAndServ)):
            raise TypeError("BlockGuardServ takes a ListenAndServ")
        super(BlockGuardServ, self).__init__(server.helper.main_program)
        self.server = server

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        self.server.complete_op()
        return super(BlockGuardServ, self).__exit__(exc_type, exc_val, exc_tb)


class ListenAndServ(object):
    """
    ListenAndServ class.

    ListenAndServ class is used to wrap listen_and_serv op to create a server
    which can receive variables from clients and run a block.
    """

    def __init__(self, endpoint, inputs, fan_in=1, optimizer_mode=True):
        self.helper = LayerHelper("listen_and_serv")
        self.inputs = inputs
        self.outputs = []
        self.endpoint = endpoint
        self.fan_in = fan_in
        # FIXME(typhoonzero): add optimizer_mode is stupid, should make it more
        # general.
        self.optimizer_mode = optimizer_mode

    def do(self):
        return BlockGuardServ(self)

    def get_params_and_grads(self):
        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()
        # params and grads in the same order.
        params = list()
        grads = list()
        for op in current_block.ops:
            # FIXME(typhoonzero): op.inputs is None if it's cloned.
            if self.optimizer_mode:
                if "Grad" in op.inputs and "Param" in op.inputs:
                    params.append(op.inputs["Param"].name)
                    grads.append(op.inputs["Grad"].name)
            else:
                # simple recv mode, recv operators inputs.
                for iname in op.input_names:
                    for in_var_name in op.input(iname):
                        params.append(parent_block.var(in_var_name))
                        grads.append(parent_block.var(in_var_name))

        return params, grads

    def parent_block(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)
        return parent_block

    def complete_op(self):
        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()
        empty_block = Program().global_block()

        parent_block.append_op(
            type='listen_and_serv',
            inputs={"X": self.inputs},
            outputs={},
            attrs={
                'endpoint': self.endpoint,
                'Fanin': self.fan_in,
                'OptimizeBlock': current_block,
                'PrefetchBlock': empty_block,
                'sync_mode': True,  # did not support async now in layers
                'grad_to_block_id': [""]
            })


def Send(endpoints, send_vars, get_vars=None):
    """
    Send layer

    Args:
        endpoints: comma seperated IP:PORT pairs in the order
                   of send_vars to send
        send_vars: vars to send
        get_vars: vars to get from server after send completes.

    Send variables to the server side, and get vars from server
    side when server have finished running server side program.
    """
    assert (type(send_vars) == list)

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Send", **locals())
    rpc_client_var = default_main_program().global_block().create_var(
        name="RPC_CLIENT_VAR", persistable=True, type=core.VarDesc.VarType.RAW)
    if not get_vars:
        get_vars = []
        for s in send_vars:
            v = helper.create_tmp_variable(dtype=s.dtype, stop_gradient=True)
            get_vars.append(v)

    helper.append_op(
        type="send",
        inputs={"X": send_vars},
        outputs={"Out": get_vars,
                 "RPCClient": rpc_client_var},
        attrs={"endpoints": endpoints,
               "epmap": epmap})
    return get_vars


def Recv(endpoints, get_vars):
    """
    Recv layer

    Args:
        endpoints: comma seperated IP:PORT pairs in the order
                   of send_vars to send
        send_vars: vars to send
        get_vars: vars to get from server after send completes.

    Send variables to the server side, and get vars from server
    side when server have finished running server side program.
    """
    assert (type(send_vars) == list)
    assert (type(get_vars) == list)

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Recv", **locals())
    helper.append_op(
        type="recv",
        inputs={"X": get_vars},
        outputs={"Out": get_vars},
        attrs={"endpoints": endpoints,
               "epmap": epmap})


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


def _copy_reader_create_op_(block, op):
    input_param_names = op.input_names
    new_input_map = {}
    for param_name in input_param_names:
        new_input_map[param_name] = []
        arg_names = op.input(param_name)
        for arg_name in arg_names:
            new_input_map[param_name].append(block.var(arg_name))

    output_param_names = op.output_names
    new_output_map = {}
    for param_name in output_param_names:
        new_output_map[param_name] = []
        arg_names = op.output(param_name)
        for arg_name in arg_names:
            new_output_map[param_name].append(block.var(arg_name))

    new_op = block.append_op(
        type=op.type,
        inputs=new_input_map,
        outputs=new_output_map,
        attrs=op.all_attrs())
    return new_op


def open_recordio_file(filename,
                       shapes,
                       lod_levels,
                       dtypes,
                       pass_num=1,
                       for_parallel=True):
    """
    Open a RecordIO file

    This layer takes a RecordIO file to read from and returns a Reader Variable.
    Via the Reader Variable, we can get data from the given RecordIO file.

    Args:
       filename(str): The RecordIO file's name.
       shapes(list): List of tuples which declaring data shapes.
       lod_levels(list): List of ints which declaring data lod_level.
       dtypes(list): List of strs which declaring data type.
       pass_num(int): Number of passes to run.
       for_parallel(Bool): Set it as True if you are going to run
            subsequent operators in parallel.

    Returns:
       Variable: A Reader Variable via which we can get RecordIO file data.

    Examples:
       .. code-block:: python

         reader = fluid.layers.io.open_recordio_file(
                                          filename='./data.recordio',
                                          shapes=[(3,224,224), (1)],
                                          lod_levels=[0, 0],
                                          dtypes=['float32', 'int64'])

         # Via the reader, we can use 'read_file' layer to get data:
         image, label = fluid.layers.read_file(reader)
    """
    dtypes = [convert_np_dtype_to_dtype_(dt) for dt in dtypes]
    shape_concat = []
    ranks = []

    for shape in shapes:
        shape_concat.extend(shape)
        ranks.append(len(shape))

    var_name = unique_name('open_recordio_file')

    startup_blk = default_startup_program().current_block()
    startup_var = startup_blk.create_var(name=var_name)
    startup_blk.append_op(
        type='create_recordio_file_reader',
        outputs={'Out': [startup_var]},
        attrs={
            'shape_concat': shape_concat,
            'lod_levels': lod_levels,
            'filename': filename,
            'ranks': ranks
        })

    startup_var.desc.set_dtypes(dtypes)
    startup_var.persistable = True
    main_prog_var = _copy_reader_var_(default_main_program().current_block(),
                                      startup_var)

    if pass_num > 1:
        main_prog_var = multi_pass(reader=main_prog_var, pass_num=pass_num)

    if for_parallel:
        main_prog_var = parallel(reader=main_prog_var)

    return monkey_patch_reader_methods(main_prog_var)


def open_files(filenames,
               shapes,
               lod_levels,
               dtypes,
               thread_num,
               buffer_size=None,
               pass_num=1,
               for_parallel=True):
    """
    Open files

    This layer takes a list of files to read from and returns a Reader Variable. 
    Via the Reader Variable, we can get data from given files. All files must 
    have name suffixs to indicate their formats, e.g., '*.recordio'. 

    Args:
       filenames(list): The list of file names.
       shapes(list): List of tuples which declaring data shapes.
       lod_levels(list): List of ints which declaring data lod_level.
       dtypes(list): List of strs which declaring data type.
       thread_num(int): The maximal concurrent prefetch thread number.
       buffer_size(int): The size of prefetch buffer.
       pass_num(int): Number of passes to run.
       for_parallel(Bool): Set it as True if you are going to run 
            subsequent operators in parallel.

    Returns:
       Variable: A Reader Variable via which we can get file data.

    Examples:
       .. code-block:: python

         reader = fluid.layers.io.open_files(filenames=['./data1.recordio',
                                                     './data2.recordio'],
                                             shapes=[(3,224,224), (1)],
                                             lod_levels=[0, 0],
                                             dtypes=['float32', 'int64'],
                                             thread_num=2,
                                             buffer_size=2)

         # Via the reader, we can use 'read_file' layer to get data:
         image, label = fluid.layers.io.read_file(reader)
    """
    if buffer_size is None:
        buffer_size = thread_num
    if isinstance(filenames, basestring):
        filenames = [filenames]
    dtypes = [convert_np_dtype_to_dtype_(dt) for dt in dtypes]
    shape_concat = []
    ranks = []

    for shape in shapes:
        shape_concat.extend(shape)
        ranks.append(len(shape))

    multi_file_reader_name = unique_name('multi_file_reader')
    startup_blk = default_startup_program().current_block()
    startup_reader = startup_blk.create_var(name=multi_file_reader_name)
    startup_blk.append_op(
        type='open_files',
        outputs={'Out': [startup_reader]},
        attrs={
            'shape_concat': shape_concat,
            'lod_levels': lod_levels,
            'ranks': ranks,
            'file_names': filenames,
            'thread_num': thread_num,
            'buffer_size': buffer_size
        })

    startup_reader.desc.set_dtypes(dtypes)
    startup_reader.persistable = True
    main_prog_reader = _copy_reader_var_(default_main_program().current_block(),
                                         startup_reader)
    if pass_num > 1:
        main_prog_reader = multi_pass(
            reader=main_prog_reader, pass_num=pass_num)

    if for_parallel:
        main_prog_reader = parallel(reader=main_prog_reader)

    return monkey_patch_reader_methods(main_prog_reader)


def __create_shared_decorated_reader__(op_type, reader, attrs):
    var_name = unique_name(op_type)
    startup_blk = default_startup_program().current_block()
    startup_var = startup_blk.create_var(name=var_name)
    startop_op = startup_blk.append_op(
        type=op_type,
        inputs={'UnderlyingReader': reader},
        outputs={'Out': [startup_var]},
        attrs=attrs)
    startup_var.persistable = True
    main_prog_block = default_main_program().current_block()
    main_prog_var = _copy_reader_var_(main_prog_block, startup_var)
    _copy_reader_create_op_(main_prog_block, startop_op)
    return monkey_patch_reader_methods(main_prog_var)


def __create_unshared_decorated_reader__(op_type, reader, attrs, name=None):
    new_reader_name = name if name is not None else unique_name(op_type)
    main_blk = default_main_program().current_block()
    new_reader = main_blk.create_var(name=new_reader_name)
    main_blk.append_op(
        type=op_type,
        inputs={'UnderlyingReader': reader},
        outputs={'Out': [new_reader]},
        attrs=attrs)
    new_reader.persistable = True
    new_reader.stop_gradient = True
    return monkey_patch_reader_methods(new_reader)


def shuffle(reader, buffer_size):
    return __create_unshared_decorated_reader__(
        'create_shuffle_reader', reader, {'buffer_size': int(buffer_size)})


def batch(reader, batch_size):
    return __create_unshared_decorated_reader__(
        'create_batch_reader', reader, {'batch_size': int(batch_size)})


def double_buffer(reader, place=None, name=None):
    attrs = dict()
    if place is not None:
        attrs['place'] = str(place).upper()
    return __create_unshared_decorated_reader__(
        'create_double_buffer_reader', reader, attrs, name=name)


def multi_pass(reader, pass_num):
    return __create_shared_decorated_reader__(
        'create_multi_pass_reader', reader, {'pass_num': int(pass_num)})


def parallel(reader):
    return __create_shared_decorated_reader__('create_threaded_reader', reader,
                                              {})


def read_file(file_obj):
    helper = LayerHelper('read_file')
    out = [
        helper.create_tmp_variable(
            stop_gradient=True, dtype='float32')
        for _ in range(len(file_obj.desc.shapes()))
    ]
    helper.append_op(
        type='read', inputs={'Reader': [file_obj]}, outputs={'Out': out})
    if len(out) == 1:
        return out[0]
    else:
        return out
