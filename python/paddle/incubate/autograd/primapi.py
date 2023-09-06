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
# limitations under the License.

import logging
import typing

import paddle
from paddle.base import core, framework
from paddle.base.core import prim_config
from paddle.incubate.autograd import primx


@framework.static_only
def to_prim(
    blocks,
    blacklist=frozenset(),
    whitelist=frozenset(),
    start_idx=-1,
    backward_length=-1,
):
    """Search nonbasic ops which have be registered composite rules and replace them with primitive ops.
    The operators in blacklist will be excluded from program when lowering into primitives, and only the
    operators in whitelist will be lowering. The priority of blacklist is higher than whitelist, it means
    an operator both in blacklist and whitelist will not be lowering.

    The finally set that will be lowering is:
        (blocks.ops & ops have decomposite rule & whitelist) - blacklist

    Args:
        blacklist(frozenset): The Operators that will be exclude when lowering into primitives.
        whitelist(frozenset): Only the operators in whitelist will be lowering into primitives.
        start_idx(int): If start_idx exceeds -1, ops[start_idx:] will be processed. Default: -1.
        backward_length(int): If backward_length exceeds -1, ops[:-backward_length] will be processed. Default: -1.
    """
    if not core._is_fwd_prim_enabled():
        return
    if isinstance(blocks, paddle.base.framework.Block):
        logging.info("Atomize composite op to primitive ops begin.")
        main_program = blocks.program
    elif isinstance(blocks, typing.Sequence):
        for item in blocks:
            if not isinstance(item, paddle.base.framework.Block):
                raise TypeError(
                    f"Expect block or sequence of blocks, but sequence contains {type(item)}."
                )
        main_program = blocks[0].program
    else:
        raise TypeError(
            f"Expect block or sequence of blocks, but got {type(blocks)}."
        )
    if not isinstance(blacklist, (set, frozenset)):
        raise TypeError(
            f'Expected type of blacklisst is set|frozenset, but got {type(blacklist)}.'
        )
    if not isinstance(whitelist, (set, frozenset)):
        raise TypeError(
            f'Expected type of whiltelist is set|frozenset, but got {type(whitelist)}.'
        )

    blacklist = prim_config["forward_blacklist"] | blacklist

    with framework.program_guard(main_program):
        print("Lowering composite forward ops begin...", flush=True)

        if len(blacklist) > 0 and len(whitelist) > 0:
            filter_ = lambda x: x.type in whitelist and x.type not in blacklist
        elif len(blacklist) > 0 and len(whitelist) == 0:
            filter_ = lambda x: x.type not in blacklist
        elif len(blacklist) == 0 and len(whitelist) > 0:
            filter_ = lambda x: x.type in whitelist
        else:
            filter_ = lambda x: True
        primx._lower_composite(
            blocks,
            filter_,
            start_idx=start_idx,
            backward_length=backward_length,
        )
        replace_ops = prim_config["composite_ops_record"]
        print(
            f"Lowering composite forward ops finish: {replace_ops}", flush=True
        )
