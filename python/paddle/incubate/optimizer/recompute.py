# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import core, framework, unique_name
from paddle.base.backward import append_backward
from paddle.base.framework import Variable, in_dygraph_mode, program_guard
from paddle.optimizer import Optimizer


class RecomputeOptimizer(Optimizer):
    """
        :api_attr: Static Graph

    Recompute Optimizer Wrapper

    Normally, a training step contains three sub-steps: first, run forward
    Operators to calculate the loss; second, run backward Operators to
    calculate gradient of the parameters; third, apply optimization method
    to update the value of the parameters.

    In the forward computation process, all variables that are needed by
    backward computation process will be kept in memory, which occupy a great
    amount of memory when the network becomes very deep.

    Recompute split the network to k segments. In each segment, It will
    recompute the forward Operators, before running backward operators. It is
    very helpful for saving memory.

    The Variables that separate a network to segments are called as checkpoints,
    and users should set it manually. The usage is very simple:

    Args:
        optimizer (Optimizer): The optimizer that is applied to parameters.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import numpy as np

            >>> paddle.enable_static()

            >>> def gen_data():
            ...     return {"x": np.random.random(size=(32, 32)).astype('float32'),
            ...     "y": np.random.randint(2, size=(32, 1)).astype('int64')}
            >>> def mlp(input_x, input_y, hid_dim=128, label_dim=2):
            ...     print(input_x)
            ...     fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
            ...     prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
            ...     cost = paddle.nn.functional.cross_entropy(
            ...         input=prediction, label=input_y,
            ...         reduction='none', use_softmax=False
            ...     )
            ...     sum_cost = paddle.mean(cost)
            ...     return sum_cost, fc_1, prediction
            >>> input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
            >>> input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
            >>> cost, fc_1, pred = mlp(input_x, input_y)

            >>> sgd = paddle.optimizer.Adam(learning_rate=0.01)
            >>> sgd = paddle.incubate.optimizer.RecomputeOptimizer(sgd)
            >>> sgd._set_checkpoints([fc_1, pred])
            >>> sgd.minimize(cost)

            >>> print("Finished optimize")
            Finished optimize
            >>> place = paddle.CPUPlace()
            >>> exe = paddle.static.Executor(place)
            >>> exe.run(paddle.static.default_startup_program())
            >>> step = 10

            >>> for i in range(step):
            ...     cost_val = exe.run(feed=gen_data(),
            ...             program=paddle.static.default_main_program(),
            ...             fetch_list=[cost.name])
            ...     print("step=%d cost=%f" % (i, cost_val[0]))
            var x : LOD_TENSOR.shape(-1, 32).dtype(float32).stop_gradient(True)
            Finished optimize
            step=0 cost=0.737203
            step=1 cost=1.308077
            step=2 cost=0.768422
            step=3 cost=1.239475
            step=4 cost=0.882643
            step=5 cost=0.738027
            step=6 cost=0.819374
            step=7 cost=0.818534
            step=8 cost=0.753692
            step=9 cost=0.787448

    """

    def __init__(self, optimizer):
        if in_dygraph_mode():
            raise Exception("In dygraph, don't support RecomputeOptimizer.")
        self._optimizer = optimizer
        self._checkpoints = None
        self._learning_rate = self._optimizer._learning_rate
        self._learning_rate_map = self._optimizer._learning_rate_map
        self.enable_offload = False

    def _set_checkpoints(self, checkpoints):
        """
        Args:
            checkpoints (list): List of Variable or string
        """
        assert isinstance(
            checkpoints, list
        ), "_checkpoints should be a list of Variable or a list of String"
        for ckpt in checkpoints:
            assert isinstance(
                ckpt, (Variable, str)
            ), "_checkpoints should be a list of Variable or a list of String"
        self._checkpoints = checkpoints

    # should enable offload before calling backward
    def _enable_offload(self):
        self.enable_offload = True

    @framework.deprecate_stat_dict
    def load(self, state_dict):
        """
            :api_attr: Static Graph

        load function is not supported by Recompute Optimizer for now.
        :return: None

        Args:
            state_dict: the dict load by load_persistable method

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()
                >>> def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                ...     fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                ...     prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                ...     cost = paddle.nn.functional.cross_entropy(
                ...         input=prediction, label=input_y,
                ...         reduction='none', use_softmax=False
                ...     )
                ...     sum_cost = paddle.mean(cost)
                ...     return sum_cost, fc_1, prediction

                >>> input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
                >>> input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
                >>> cost, fc_1, pred = mlp(input_x, input_y)
                >>> print("Finished FF")
                Finished FF

                >>> sgd = paddle.optimizer.Adam(learning_rate=0.01)
                >>> sgd = paddle.incubate.optimizer.RecomputeOptimizer(sgd)
                >>> sgd._set_checkpoints([fc_1, pred])
                >>> try:
                ...     state_dict = {}
                ...     sgd.load(state_dict)
                >>> except NotImplementedError as e:
                ...     print(e)
                load function is not supported by Recompute Optimizer for now
        """
        raise NotImplementedError(
            "load function is not supported by Recompute Optimizer for now"
        )

    def apply_gradients(self, params_grads):
        """
        call apply_gradients function of self._optimizer.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.base.framework as framework

                >>> paddle.enable_static()

                >>> def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                ...     fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                ...     prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                ...     cost = paddle.nn.functional.cross_entropy(
                ...         input=prediction, label=input_y,
                ...         reduction='none', use_softmax=False
                ...     )
                ...     sum_cost = paddle.mean(cost)
                ...     return sum_cost, fc_1, prediction

                >>> input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
                >>> input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
                >>> cost, fc_1, pred = mlp(input_x, input_y)
                >>> print("Finished FF")
                Finished FF

                >>> sgd = paddle.optimizer.Adam(learning_rate=0.01)
                >>> sgd = paddle.incubate.optimizer.RecomputeOptimizer(sgd)
                >>> sgd._set_checkpoints([fc_1, pred])
                >>> params_grads = sgd.backward(
                ...     cost,
                ...     startup_program=None,
                ...     parameter_list=None,
                ...     no_grad_set=None)

                >>> program = cost.block.program
                >>> with framework.program_guard(program, None):
                ...     optimize_ops = sgd.apply_gradients(params_grads)

                >>> print("Finished apply gradients")
                Finished apply gradients
        """

        return self._optimizer.apply_gradients(params_grads=params_grads)

    def _create_vars(self, varname):
        pinned_var_name = unique_name.generate(varname + "@Pinned")
        fetched_var_name = unique_name.generate(varname + "@Fetch")

        pinned_var = self._main_program.global_block().create_var(
            name=pinned_var_name,
            shape=self.checkpoint_shape,
            dtype=self._main_program.global_block().var(varname).dtype,
            persistable=False,
            stop_gradient=True,
        )

        fetch_var = self._main_program.global_block().create_var(
            name=fetched_var_name,
            shape=self.checkpoint_shape,
            dtype=self._main_program.global_block().var(varname).dtype,
            persistable=False,
            stop_gradient=False,
        )

        return pinned_var_name, fetched_var_name

    def _append_fill_constant_ops(self, startup_program):
        """
        add fill_constant_ops to the end of the prog

        we should fill the pinned vars before running the main_prog
        to instantiate their tensor hold_, which could tell us whether
        the host memory could hold all the checkpoints from all the
        GPU devices in this node.
        """
        op_role = 0
        block = startup_program.global_block()
        fill_constant_vars = self.checkpoint_name2pinned_name.values()
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
        for varname in fill_constant_vars:
            var = self._main_program.global_block().var(varname)
            # NOTE (JZ-LIANG) to pre-allocate the CUDAPinned MEM
            pinned_var = block.create_var(
                name=varname,
                shape=self.checkpoint_shape,
                dtype=self._main_program.global_block().var(var.name).dtype,
                persistable=False,
                stop_gradient=True,
            )
            block.append_op(
                type='fill_constant',
                outputs={'Out': varname},
                attrs={
                    "shape": var.shape,
                    "dtype": var.dtype,
                    "value": 0.0,
                    "place_type": 2,
                    OP_ROLE_KEY: op_role,
                },
            )

    def _insert_async_memcpy_op(
        self, insert_idx, src_varname, dst_varname, op_role, dst_place_type
    ):
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
        self.block._insert_op_without_sync(
            insert_idx,
            type='memcpy',
            inputs={'X': [self._main_program.global_block().var(src_varname)]},
            outputs={
                'Out': [self._main_program.global_block().var(dst_varname)]
            },
            attrs={"dst_place_type": int(dst_place_type), OP_ROLE_KEY: op_role},
        )

    def _insert_fetch_op(self, idx, varname):
        assert (
            varname in self.checkpoint_name2pinned_name
        ), f"Try to fetch {varname} from Pinned Memory, but it is NOT a checkpoint"

        pinned_varname = self.checkpoint_name2pinned_name[varname]
        fetch_varname = self.checkpoint_name2fetch_name[varname]
        self._insert_async_memcpy_op(idx, pinned_varname, fetch_varname, 1, 1)

    def _insert_offload_op(self, idx, varname):
        assert (
            varname in self.checkpoint_name2pinned_name
        ), f"Try to offload {varname} to Pinned Memory, but it is NOT a checkpoint"
        pinned_varname = self.checkpoint_name2pinned_name[varname]
        self._insert_async_memcpy_op(idx, varname, pinned_varname, 0, 2)

    def _insert_sync_op(self, op_idx, checkpoint_name):
        # single stream offload no need sync
        pass

    def _record_fetch_op(self, idx):
        assert (
            len(self.un_fetch_checkpoint_names) > 0
        ), "Could NOT found checkpoint to fetch"
        checkpoint_name = self.un_fetch_checkpoint_names.pop(-1)
        logging.debug(f"Record fetch [{checkpoint_name}]")
        self.idx2insertions[idx] = ("fetch", checkpoint_name)

        return checkpoint_name

    def _record_offload_op(self, idx, checkpoint_name):
        expected_checkpoint_name = self.un_offload_checkpoint_names.pop(0)
        assert (
            checkpoint_name == expected_checkpoint_name
        ), f"expected to offload [{expected_checkpoint_name}] but got [{checkpoint_name}]"
        logging.debug(f"Record offload [{checkpoint_name}]")
        self.idx2insertions[idx] = ("offload", checkpoint_name)

    def _record_sync_op(self, idx, checkpoint_name):
        assert (
            checkpoint_name not in self.synced_checkpoints
        ), f"Try to sync the checkpoint [{checkpoint_name}] twice"
        self.synced_checkpoints.add(checkpoint_name)
        logging.debug(f"Record offload sync [{checkpoint_name}]")
        self.idx2insertions[idx] = ("sync", checkpoint_name)

    def _parse_backward(self):
        self.idx2insertions = {}
        # don't offload the last checkpoints, to favor throughput
        self.un_fetch_checkpoint_names = self.sorted_checkpoint_names[:]
        self.un_fetch_checkpoint_names.pop(-1)
        need_fetch_checkpoint_names = self.un_fetch_checkpoint_names[:]
        self.checkpoint_usage_count = {}
        for checkpoint_name in self.un_fetch_checkpoint_names:
            self.checkpoint_usage_count[checkpoint_name] = 0

        self.bw_start_op_idx = len(self.block.ops)
        for idx, op in enumerate(self.block.ops):
            if int(op.desc.attr("op_role")) == 1:
                self.bw_start_op_idx = idx
                break

        assert self.bw_start_op_idx < len(
            self.block.ops
        ), "Could NOT found backward op in prog"

        # fetch second to last checkpoint at the beginning of BW
        fetched_checkpoint_varname = self._record_fetch_op(self.bw_start_op_idx)
        last_last_fetch_checkpoint = None

        for i, op in enumerate(self.block.ops[self.bw_start_op_idx :]):
            idx = self.bw_start_op_idx + i
            input_vars = op.desc.input_arg_names()

            for input_var in input_vars:
                if input_var in need_fetch_checkpoint_names:
                    if input_var not in self.un_fetch_checkpoint_names:
                        # fetch the offload checkpoint when the first usage of its previous one
                        if self.checkpoint_usage_count[input_var] == 0:
                            # TODO (JZ-LIANG) sync memcpy_stream if extra stream for memcpy
                            second_to_last_fetch_checkpoint = (
                                fetched_checkpoint_varname
                            )
                            # there is NO fetch ahead the first checkpoint
                            if input_var != self.sorted_checkpoint_names[0]:
                                fetched_checkpoint_varname = (
                                    self._record_fetch_op(idx)
                                )

                        # should check the current used checkpoint is ths last fetch one
                        assert (
                            second_to_last_fetch_checkpoint == input_var
                        ), f"Current recompute segment should use [{second_to_last_fetch_checkpoint}] BUT got [{input_var}]"
                        # rename
                        self.block.ops[idx]._rename_input(
                            input_var,
                            self.checkpoint_name2fetch_name[input_var],
                        )
                        self.checkpoint_usage_count[input_var] += 1
                    else:
                        raise ValueError(
                            f"use checkpoint [{input_var}] before fetch in BW"
                        )

        assert (
            len(self.un_fetch_checkpoint_names) == 0
        ), f"{self.un_fetch_checkpoint_names} checkpoints have NOT been Recorded"

    def _update_backward(self):
        if len(self.idx2insertions) == 0:
            return
        total_op = len(self.block.ops)
        for op_idx in reversed(range(self.bw_start_op_idx, total_op)):
            if op_idx in self.idx2insertions:
                operation, checkpoint_name = self.idx2insertions[op_idx]
                if operation == "fetch":
                    self._insert_fetch_op(op_idx, checkpoint_name)
                    logging.debug(f"Insert [{checkpoint_name}] fetch op.")
                    del self.idx2insertions[op_idx]
                elif operation == "sync":
                    self._insert_sync_op(op_idx, checkpoint_name)
                    logging.debug(f"Sync [{checkpoint_name}] fetch op.")
        self.block._sync_with_cpp()
        assert (
            len(self.idx2insertions) == 0
        ), f"{[ele[1] for ele in self.idx2insertions.values()]} checkpoints left un-Fetched"

    def _parse_forward(self):
        self.idx2insertions = {}
        # don't offload the last checkpoints, faster, less memory saving
        self.un_offload_checkpoint_names = self.sorted_checkpoint_names[:]
        last_checkpoint = self.un_offload_checkpoint_names.pop(-1)
        need_offload_checkpoint_names = self.un_offload_checkpoint_names[:]
        self.checkpoint_usage_count_and_idx = {}
        for checkpoint_name in self.un_offload_checkpoint_names:
            self.checkpoint_usage_count_and_idx[checkpoint_name] = {
                'count': 0,
                'idx': -1,
            }
        self.synced_checkpoints = set()
        self.fw_start_op_idx = len(self.block.ops)
        for idx, op in enumerate(self.block.ops):
            if int(op.desc.attr("op_role")) == 0:
                self.fw_start_op_idx = idx
                break

        assert self.fw_start_op_idx < len(
            self.block.ops
        ), "Could NOT found Forward op in prog"
        last_offload_checkpoint = None

        for i, op in enumerate(
            self.block.ops[self.fw_start_op_idx : self.bw_start_op_idx]
        ):
            idx = self.fw_start_op_idx + i
            output_vars = op.desc.output_arg_names()
            input_vars = op.desc.input_arg_names()

            for output_var in output_vars:
                if output_var in need_offload_checkpoint_names:
                    assert (
                        len(output_vars) == 1
                    ), f"checkpoint should be the only Output of a certain op, but [{output_var}] is from [{op}]"

                    if output_var in self.un_offload_checkpoint_names:
                        # insert sync op if last checkpoint has not been sync
                        if last_offload_checkpoint is not None:
                            if (
                                self.checkpoint_usage_count_and_idx[
                                    last_offload_checkpoint
                                ]['count']
                                == 0
                            ):
                                self._record_sync_op(
                                    idx, last_offload_checkpoint
                                )
                            else:
                                last_usage_idx = (
                                    self.checkpoint_usage_count_and_idx[
                                        last_offload_checkpoint
                                    ]['idx']
                                )
                                assert (
                                    last_usage_idx > 0
                                ), f"last_usage_idx of checkpoint [{last_offload_checkpoint}] should large than 0"
                                self._record_sync_op(
                                    last_usage_idx + 1, last_offload_checkpoint
                                )
                        # insert offload op after the checkpoint's generation op
                        self._record_offload_op(idx + 1, output_var)
                        last_offload_checkpoint = output_var
                    else:
                        raise ValueError(
                            f"There should be just ONE op that output checkpoint [{output_var}]"
                        )
                # need to sync the last need to offload checkpoint before the last checkpoint as output op
                if output_var == last_checkpoint:
                    assert (
                        len(output_vars) == 1
                    ), f"checkpoint should be the only Output of a certain op, but [{output_var}] is from [{op}]"
                    assert (
                        last_offload_checkpoint
                        == self.sorted_checkpoint_names[-2]
                    ), f"the last offload checkpoint before [{last_checkpoint}] is suppose to be [{self.sorted_checkpoint_names[-2]}], but got [{last_offload_checkpoint}]"
                    # sync if last checkpoint has not been sync
                    if (
                        self.checkpoint_usage_count_and_idx[
                            last_offload_checkpoint
                        ]['idx']
                        == 0
                    ):
                        self._record_sync_op(idx, last_offload_checkpoint)
                    else:
                        last_usage_idx = self.checkpoint_usage_count_and_idx[
                            last_offload_checkpoint
                        ]['idx']
                        assert (
                            last_usage_idx > 0
                        ), f"last_usage_idx of checkpoint [{last_offload_checkpoint}] should large than 0"
                        self._record_sync_op(
                            last_usage_idx + 1, last_offload_checkpoint
                        )
            # record checkpoint usage
            for input_var in input_vars:
                if input_var in need_offload_checkpoint_names:
                    assert (
                        input_var not in self.synced_checkpoints
                    ), f"checkpoint [{input_var}] used after sync"
                    self.checkpoint_usage_count_and_idx[input_var]['count'] += 1
                    self.checkpoint_usage_count_and_idx[input_var]['idx'] = idx

        assert (
            len(self.un_offload_checkpoint_names) == 0
        ), f"{self.un_fetch_checkpoint_names} checkpoints have NOT been Recorded"
        assert len(self.synced_checkpoints) == len(
            need_offload_checkpoint_names
        ), f"{set(need_offload_checkpoint_names) - set(self.synced_checkpoints)} checkpoints have NOT been Recorded"

    def _update_forward(self):
        if len(self.idx2insertions) == 0:
            return
        for op_idx in reversed(
            range(self.fw_start_op_idx, self.bw_start_op_idx)
        ):
            if op_idx in self.idx2insertions:
                operation, checkpoint_name = self.idx2insertions[op_idx]
                if operation == "offload":
                    self._insert_offload_op(op_idx, checkpoint_name)
                    logging.debug(f"Insert [{checkpoint_name}] offload op.")
                    del self.idx2insertions[op_idx]
                elif operation == "sync":
                    self._insert_sync_op(op_idx, checkpoint_name)
                    logging.debug(
                        f"Insert [{checkpoint_name}] offload_sync op."
                    )
                    del self.idx2insertions[op_idx]

        self.block._sync_with_cpp()
        assert (
            len(self.idx2insertions) == 0
        ), f"{[ele[1] for ele in self.idx2insertions.values()]} checkpoints left un-Offloaded"

    def _check_offload_fetch(self):
        # TODO(JZ-LIANG) the single stream offload need no sync
        pass

    def _offload(self, loss, startup_program=None):
        """
        core steps for recompute offload
        1. create pinned vars and temp vars
        2. parse & update Forward pass: offload, sync
        3. parse & update Backward pass: rename, fetch, sync
        4. verify the correctness
        """
        self._main_program = loss.block.program
        self.block = loss.block
        if startup_program is None:
            startup_program = paddle.static.default_startup_program()

        with program_guard(self._main_program, startup_program):
            assert (
                len(self.checkpoint_shape) > 0
            ), f"checkpoints shape {self.checkpoint_shape} should be an non empty list like: [12, 512, 1024]"
            assert all(
                ele > 0 for ele in self.checkpoint_shape
            ), f"all ele in checkpoints shape {self.checkpoint_shape} should be a determined integer larger than 0"
            self.checkpoint_name2pinned_name = {}
            self.checkpoint_name2fetch_name = {}
            for checkpoint_varname in self.sorted_checkpoint_names:
                pinned_var_name, fetch_var_name = self._create_vars(
                    checkpoint_varname
                )
                self.checkpoint_name2pinned_name[
                    checkpoint_varname
                ] = pinned_var_name
                self.checkpoint_name2fetch_name[
                    checkpoint_varname
                ] = fetch_var_name
            self._append_fill_constant_ops(startup_program)
            # TODO (JZ-LIANG) to provide two offload strategy in future
            # step 2. parse & update FW: rename, offload, sync
            self._parse_backward()
            self._update_backward()
            # step 3. parse & update BW: rename, offload, sync
            self._parse_forward()
            self._update_forward()
            # step 4. verify the correctness
            self._check_offload_fetch()

    def backward(
        self,
        loss,
        startup_program=None,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None,
    ):
        """
        call append_backward with checkpoints.

        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables or Variable.names to update.
            no_grad_set (set|None): set of Variables or Variables.names should be ignored.
            callbacks (list|None): list of callables to run when appending backward
                operator for one parameter.
            checkpoints (list): list of Variables as checkpoints

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                ...     fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                ...     prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                ...     cost = paddle.nn.functional.cross_entropy(
                ...         input=prediction, label=input_y,
                ...         reduction='none', use_softmax=False
                ...     )
                ...     sum_cost = paddle.mean(cost)
                ...     return sum_cost, fc_1, prediction

                >>> input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
                >>> input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
                >>> cost, fc_1, pred = mlp(input_x, input_y)
                >>> print("Finished FF")
                Finished FF

                >>> sgd = paddle.optimizer.Adam(learning_rate=0.01)
                >>> sgd = paddle.incubate.optimizer.RecomputeOptimizer(sgd)
                >>> sgd._set_checkpoints([fc_1, pred])
                >>> params_grads = sgd.backward(
                ...     cost,
                ...     startup_program=None,
                ...     parameter_list=None,
                ...     no_grad_set=None)
                >>> print("Finished backward")
                Finished backward
        """
        assert (
            self._checkpoints is not None
        ), "You should call _set_checkpoints first"

        if in_dygraph_mode():
            raise NotImplementedError(
                "DyGraph current does not support recompute"
            )

        self._dtype = loss.dtype
        program = loss.block.program
        with program_guard(program, startup_program):
            checkpoint_vars = []
            for ckpt in self._checkpoints:
                if isinstance(ckpt, Variable):
                    checkpoint_vars.append(ckpt)
                else:
                    checkpoint_vars.append(loss.block.var(ckpt))

            # allow return to non-recompute when checkpoints is empty
            if len(checkpoint_vars) > 0:
                params_grads, sorted_checkpoint_names = append_backward(
                    loss,
                    parameter_list,
                    no_grad_set,
                    checkpoints=checkpoint_vars,
                )
            else:
                params_grads = append_backward(
                    loss,
                    parameter_list,
                    no_grad_set,
                    checkpoints=checkpoint_vars,
                )

        if self.enable_offload:
            self.sorted_checkpoint_names = sorted_checkpoint_names
            self._offload(loss, startup_program=startup_program)

        return params_grads

    def apply_optimize(self, loss, startup_program, params_grads):
        """
        call the apply_optimize function of self._optimizer
        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            params_grads (list): list of (param, grad) pair to do optimization.
        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()

                >>> def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                ...     fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                ...     prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                ...     cost = paddle.nn.functional.cross_entropy(
                ...         input=prediction, label=input_y,
                ...         reduction='none', use_softmax=False
                ...     )
                ...     sum_cost = paddle.mean(cost)
                ...     return sum_cost, fc_1, prediction

                >>> input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
                >>> input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
                >>> cost, fc_1, pred = mlp(input_x, input_y)
                >>> print("Finished FF")
                Finished FF

                >>> sgd = paddle.optimizer.Adam(learning_rate=0.01)
                >>> sgd = paddle.incubate.optimizer.RecomputeOptimizer(sgd)
                >>> sgd._set_checkpoints([fc_1, pred])
                >>> params_grads = sgd.backward(
                ...     cost,
                ...     startup_program=None,
                ...     parameter_list=None,
                ...     no_grad_set=None)

                >>> optimize_ops = sgd.apply_optimize(
                ...     cost, startup_program=None, params_grads=params_grads)

                >>> print("Finished apply_optimize")
                Finished apply_optimize
        """

        func = (
            self._optimizer.apply_optimize
            if hasattr(self._optimizer, 'apply_optimize')
            else self._optimizer._apply_optimize
        )
        return func(
            loss, startup_program=startup_program, params_grads=params_grads
        )

    def minimize(
        self, loss, startup_program=None, parameter_list=None, no_grad_set=None
    ):
        assert isinstance(loss, Variable), "The loss should be an Variable."
        assert (
            self._checkpoints is not None
        ), "You should call _set_checkpoints first"
        if in_dygraph_mode():
            raise NotImplementedError(
                "DyGraph current does not support recompute"
            )
        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set,
        )

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads
        )

        return optimize_ops, params_grads
