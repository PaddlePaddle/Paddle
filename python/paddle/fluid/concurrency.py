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

# TODO: Variables: make_channel
# TODO: Operators: send, close_channel, recv, go, select
from layers.control_flow import BlockGuard
from layer_helper import LayerHelper
import core
__all__ = [
    'Go',
    'make_channel',
    'channel_send',
    'channel_recv',
    'channel_close',
]


class Go(BlockGuard):
    def __init__(self, name=None):
        self.helper = LayerHelper("go", name=name)
        super(Go, self).__init__(self.helper.main_program)

    def __enter__(self):
        super(Go, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.construct_go_op()
        return super(Go, self).__exit__(exc_type, exc_val, exc_tb)

    def construct_go_op(self):
        main_program = self.helper.main_program
        go_block = main_program.current_block()
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)

        x_name_list = set()
        out_vars = []
        for op in go_block.ops:
            # Iterate over all operators, get all the inputs
            # and add as input to the Go operator.
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    x_name_list.add(in_var_name)

            # Iterate over all operators , get all the outputs
            # add to the output list of Go operator only if
            # they exist in the parent block.
            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    if out_var_name in parent_block.vars:
                        out_vars.add(parent_block.var(out_var_name))

        parent_block.append_op(
            type='go',
            inputs={'X': [parent_block.var(x_name) for x_name in x_name_list]},
            outputs={'Out': out_vars},
            attrs={'sub_block': go_block})


def make_channel(dtype, capacity=0):
    """
    Helps implementation of a concurrent program by creating a "channel" of
    a defined data type. Channels allow for the passing of data in
    concurrent scenarios - such as when using threads to divide computation.
    Channels can be used to "send" and "receive" such data concurrently.

    There are two kinds of channels: unbuffered and buffered. Unbuffered
    channels have no capacity - and thus, block on send and only unblock only
    once what they have sent has been received.

    On the other hand, buffered channels are initialized with a capacity -
    and do not block on sends.

    Use this method in combination with `channel_send`, `channel_recv`,
    `channel_close`, and `Go` to design a concurrent Paddle program.

    Args:
        dtype (ParamAttr|int): Data type of the data sent in the channel.
        This data type should be one of the Paddle supported data types.
        capacity (ParamAttr|int): Size of the channel. Defaults to 0 for
        to create an unbuffered channel.

    Returns:
        Variable: The channel variable that can be used to send an receive data
                  of the defined dtype.

    Examples:
        .. code-block:: python

          ch = fluid.make_channel(dtype='int32', capacity=10)
          ...
          # Code to execute in a Go block, which receives the channel data.
          fluid.channel_send(ch, 100)
          fluid.channel_close(ch)
    """
    helper = LayerHelper('make_channel', **locals())
    main_program = helper.main_program
    make_channel_block = main_program.current_block()

    # Make a channel variable (using the channel data type) and make sure it
    # persists into the global scope.
    channel = helper.create_variable(
        dtype=core.VarDesc.VarType.CHANNEL, persistable=True)

    create_channel_op = make_channel_block.append_op(
        type="channel_create",
        outputs={"Out": channel},
        attrs={"data_type": dtype,
               "capacity": capacity})

    return create_channel_op


def channel_send(channel, value):
    """
    Sends a value through a channel variable. Used by an unbuffered or buffered
    channel to pass data from within or to a concurrent Go block, where
    `channel_recv` to used to get the passed value.

    Args:
        channel (Variable|Channel): Channel variable created using
        `make_channel`.

    Returns:
        Variable: The boolean status on whether or not the channel
                  successfully sent the passed value.

    Examples:
        .. code-block:: python

          ch = fluid.make_channel(dtype='int32', capacity=10)
          ...
          # Code to execute in a Go block, which receives the channel data.
          fluid.channel_send(ch, 100)
    """
    helper = LayerHelper('channel_send', **locals())
    main_program = helper.main_program
    channel_send_block = main_program.current_block()
    status = helper.create_variable(dtype=core.VarDesc.VarType.TENSOR)

    channel_send_op = channel_send_block.append_op(
        type="channel_send",
        inputs={
            "Channel": channel,
            "X": value,
        },
        outputs={"Status": status})

    return channel_send_op


def channel_recv(channel, dtype):
    """
    Receives a value through a channel variable. Used by an unbuffered or
    buffered channel within a concurrent Go block to get data from originally
    sent using `channel_send`, or from outside such a block where
    `channel_send` is used to send the value.

    Args:
        channel (Variable|Channel): Channel variable created using
        `make_channel`.
        dtype (Variable|int): Data type of the data expected to be read in the
        channel. This data type should be one of the Paddle supported data
        types.

    Returns:
        Variable: The boolean status on whether or not the channel
                  successfully received the passed value.

    Examples:
        .. code-block:: python

          ch = fluid.make_channel(dtype='int32', capacity=10)
          with fluid.Go():
            fluid.channel_recv(ch, 'int32')

          # Code to send data through the channel.
    """
    helper = LayerHelper('channel_recv', **locals())
    main_program = helper.main_program
    channel_recv_block = main_program.current_block()

    return_value = helper.create_variable(dtype=dtype)
    status = helper.create_variable(dtype=core.VarDesc.VarType.TENSOR)

    channel_recv_op = channel_recv_block.append_op(
        type="channel_recv",
        inputs={"Channel": channel},
        outputs={"Out": return_value,
                 "Status": status})

    return channel_recv_op


def channel_close(channel):
    """
    Closes a channel created using `make_channel`.

    Args:
        channel (Variable|Channel): Channel variable created using
        `make_channel`.

    Examples:
        .. code-block:: python

          ch = fluid.make_channel(dtype='int32', capacity=10)
          ...
          # Code to receive and send data through a channel
          ...
          fluid.channel_close(ch)
    """
    helper = LayerHelper('channel_close', **locals())
    main_program = helper.main_program
    channel_close_block = main_program.current_block()

    channel_close_op = channel_close_block.append_op(
        type="channel_close", inputs={"Channel": channel})

    return channel_close_op
