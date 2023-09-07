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
import paddle

from ....base.framework import Variable
from ....framework import LayerHelper, core


class BlockGuardServ(paddle.static.nn.control_flow.BlockGuard):
    """
    BlockGuardServ class.

    BlockGuardServ class is used to create an op with a block in a program.
    """

    def __init__(self, server):
        if not (isinstance(server, ListenAndServ)):
            raise TypeError("BlockGuardServ takes a ListenAndServ")
        super().__init__(server.helper.main_program)
        self.server = server

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        self.server.complete_op()
        return super().__exit__(exc_type, exc_val, exc_tb)


class ListenAndServ:
    """
    **ListenAndServ Layer**

    ListenAndServ is used to create a rpc server bind and listen
    on specific TCP port, this server will run the sub-block when
    received variables from clients.

    Args:
        endpoint(string): IP:port string which the server will listen on.
        inputs(list): a list of variables that the server will get from clients.
        fan_in(int): how many client are expected to report to this server, default: 1.
        optimizer_mode(bool): whether to run the server as a parameter server, default: True.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> from paddle.incubate.nn.layer.io import ListenAndServ
            >>> import paddle
            >>> paddle.enable_static()
            >>> place = paddle.CPUPlace()
            >>> main = paddle.static.Program()
            >>> with paddle.static.program_guard(main):
            ...     serv = ListenAndServ(
            ...         "127.0.0.1:6170", ["X"], optimizer_mode=False)
            ...     with serv.do():
            ...         x = paddle.static.data(
            ...             shape=[32, 32],
            ...             dtype='float32',
            ...             name="X")
            ...         paddle.nn.initializer.Constant(value=1.0)(x, main.global_block())
            ...         paddle.scale(x=x, scale=10.0)

            >>> exe = paddle.static.Executor(place)
            >>> exe.run(main)
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
        params = []
        grads = []
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
        from paddle.incubate.distributed.fleet.parameter_server.mode import (
            DistributedMode,
        )

        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()

        parent_block.append_op(
            type='listen_and_serv',
            inputs={"X": self.inputs},
            outputs={},
            attrs={
                'endpoint': self.endpoint,
                'Fanin': self.fan_in,
                'optimize_blocks': [
                    current_block
                ],  # did not support multiple optimize blocks in layers
                'distributed_mode': DistributedMode.SYNC,  # did not support async now in layers
                'grad_to_block_id': [""],
            },
        )


def Send(endpoints, send_vars, dummy_output=None, sync=True):
    """
    Send variables to the server side, and get vars from server
    side when server have finished running server side program.

    Args:
        endpoints (str): comma separated IP:PORT pairs in the order
                   of send_vars to send
        send_vars (list): variables to send to server
        sync (bool): whether to wait the request finish

    """
    assert type(send_vars) == list

    if dummy_output is None:
        dummy_output = []
    elif isinstance(dummy_output, Variable):
        dummy_output = [dummy_output]

    assert type(dummy_output) == list

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Send", **locals())
    rpc_op_role_name = core.op_proto_and_checker_maker.kOpRoleAttrName()

    helper.append_op(
        type="send",
        inputs={"X": send_vars},
        outputs={"Out": dummy_output},
        attrs={
            "endpoints": endpoints,
            "epmap": epmap,
            rpc_op_role_name: core.op_proto_and_checker_maker.OpRole.RPC,
        },
    )
    if sync:
        helper.append_op(
            type="send_barrier",
            inputs={"X": dummy_output},
            outputs={"Out": []},
            attrs={"endpoints": endpoints},
        )


def Recv(endpoints, get_vars, dummy_input=None, sync=True):
    """
    Receive variables from server side

    Args:
        endpoints (str): comma separated IP:PORT pairs in the order
                   of send_vars to send
        get_vars (list): vars to get from server after send completes.
        sync (bool): whether to wait the request finish

    Returns:
        list: list of received variables
    """
    assert type(get_vars) == list

    if dummy_input is None:
        dummy_input = []
    elif isinstance(dummy_input, Variable):
        dummy_input = [dummy_input]

    assert type(dummy_input) == list

    epmap = endpoints.split(",")
    endpoints = list(set(epmap))

    helper = LayerHelper("Recv", **locals())
    helper.append_op(
        type="recv",
        inputs={"X": dummy_input},
        outputs={"Out": get_vars},
        attrs={"endpoints": endpoints, "epmap": epmap},
    )
    if sync:
        helper.append_op(
            type="fetch_barrier",
            outputs={"Out": get_vars},
            attrs={"endpoints": endpoints},
        )
    return get_vars
