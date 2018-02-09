#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
from ..layer_helper import LayerHelper
from control_flow import BlockGuard
from ..layer_helper import LayerHelper

__all__ = ['data', 'BlockGuardServ', 'ListenAndServ', 'Send']


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

    def __init__(self, endpoint, fan_in=1, optimizer_mode=True):
        self.helper = LayerHelper("listen_and_serv")
        self.inputs = []
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

        params, grads = self.get_params_and_grads()
        param_names = [p.name for p in params]
        grad_names = [g.name for g in grads]
        parent_block.append_op(
            type='listen_and_serv',
            inputs={},
            outputs={},
            attrs={
                'endpoint': self.endpoint,
                'Fanin': self.fan_in,
                'ParamList': param_names,
                'GradList': grad_names,
                'OptimizeBlock': current_block
            })


def Send(endpoints, send_vars, get_vars):
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
    assert (type(get_vars) == list)

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Send", **locals())
    helper.append_op(
        type="send",
        inputs={"X": send_vars},
        outputs={"Out": get_vars},
        attrs={"endpoints": endpoints,
               "epmap": epmap})


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
