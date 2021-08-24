# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import framework as framework
from paddle.fluid import core, unique_name
from paddle.fluid.framework import Program, Parameter, Variable, program_guard
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.fluid.backward import append_backward, _some_in_set_, _append_grad_suffix_
from paddle.distributed.auto_parallel.operators.common import get_distributed_operator
from paddle.distributed.auto_parallel.operators.common import find_best_compatible_distributed_operator_impl
from paddle.fluid.clip import GradientClipBase, GradientClipByNorm, error_clip_callback, append_gradient_clip_ops, ClipGradByGlobalNorm
from paddle.distributed.fleet.base.distributed_strategy import DistributedStrategy
from paddle.distributed.auto_parallel.context import DistributedContext
from paddle.distributed.fleet.meta_optimizers.common import is_loss_grad_op, is_backward_op, is_optimizer_op
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY
from .process import new_process_group
from .interface import _g_process_mesh_map
from .utils import _get_comm_group

__varname_not_in_block__ = ["lod_tensor_blocking_queue_0"]


class AutoParallelTranspiler(object):
    """
    warning:: AutoParallelTranspiler is experimental and subject to change.

    Transpiler convert a program into another program.
    Given a serial program which has been auto completed with shard annotation, the Transpiler 
    convert the serial program into a "distributed" program. The Transpiler will  modify the serial
    program in following two ways, which is also the major difference between serial and distributed program:
        1. partition op: replace a serial op into its corresponding dist op infered from the shard annotation
        2. partition var: if a var is sharded, modify the shape of var according to its shard annotation

    Beside the Transpiler will rewrite the program according to the user defined strategy, like AMP / Recompute / Sharding.

    AutoParallelTranspiler is supposed to be call by the auto parallel framework, and not supposed to be used by user.
    Example:
        ....
            import paddle.distributed.auto_parallel as auto
            from paddle.fluid.distributed_attribute import get_default_distributed_context
            from paddle.distributed import fleet
            from paddle.distributed.auto_parallel.transpiler import AutoParallelTranspiler

            # create serial program with forward only 
            with static.program_guard(serial_main_program, serial_start_program):
                model = create_model(config)
                tokens = static.data(name="tokens", shape=[batch_size, sequence_len], dtype='int64')
                labels = static.data(name="labels", shape=[batch_size, sequence_len], dtype='int64')
                loss_mask = static.data(name="loss_mask", shape=[batch_size, sequence_len], dtype='int64')
                preds = model(tokens)
                loss = criterion(preds, labels, loss_mask)

            # auto completion
            auto.ProcessMesh(shape=[2, 4], process_group=[0, 1, 2, 3, 4, 5, 6, 7])
            annotated_main_program = auto.complete_annotation(serial_main_program)
            auto_paralle_context = get_default_distributed_context()
                
            # distributed strategy & rank info
            rank_id = paddle.distributed.get_rank()
            dist_strategy = fleet.DistributedStrategy()
    
            # create transpiler
            APTranspiler = AutoParallelTranspiler(dist_strategy, auto_paralle_context, rank_id)

            # create dist program with forward only
            # for distributed inference, using partitioned_main_prog from here
            partitioned_main_prog, partitioned_startup_prog = APTranspiler.transpile_forward(complete_train_program, start_program)

            # create dist program with forward/backward/update
            # for distributed training, using partitioned_main_prog from here
            dist_params_grads = APTranspiler.apply_backward(loss, complete_train_program, start_program, partitioned_main_prog, partitioned_startup_prog)
            optimizer = paddle.fluid.optimizer.AdamOptimizer(
                learning_rate=0.00001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08,
                grad_clip=None)
            opt_ops = APTranspiler.apply_optimize(optimizer, dist_params_grads, partitioned_main_prog, partitioned_startup_prog)
    """

    def __init__(self, dist_strategy, auto_parallel_context, rank_id=0):
        """
        Args:
            dist_strategy (paddle.fleet.distributed_strategy): used to determine the user defined distributed strategy, like AMP and Recompute, since these strategy will effect how the program should be transpiled.
            auto_parallel_context (paddle.fluid.DistributedContext): used to access the distributed_attr of var & op, every Transpiler object could maintain its own DistributedContext member, and partition program base on that shard scenario.
            rank_id (int): global rank id to which the partitioned distributed program belong.
        """

        if not isinstance(dist_strategy, DistributedStrategy):
            raise TypeError(
                "dist_strategy be paddle.fleet.base.DistributedStrategy, got %s here"
                % type(dist_strategy))

        if not isinstance(auto_parallel_context, DistributedContext):
            raise TypeError(
                "auto_parallel_context be paddle.fluid.DistributedContext, got %s here"
                % type(auto_parallel_context))

        self._dist_strategy = dist_strategy
        self._auto_parallel_context = auto_parallel_context
        self._rank_id = rank_id
        self._serial2dist_varname_mapping = {}
        self._dist_varname_suffix = ""

        # TODO if there is some dist op that is not compatible 
        # with auto_backward in forward, the following flag 
        # should be set to False
        self._compatible_with_auto_backward = True

        # data parallelism        
        self._enable_data_parallel = False
        self._dp_degree = 0
        self._dp_group = None

        # tensor parallelism        
        self._enable_tensor_parallel = False
        self._tp_degree = 0
        self._tp_group = None

    def transpile_forward(self, serial_main_program, serial_startup_program):
        """
        take serial forward programs with shard annotation, create a new distributed forward programs based on the serial ones.
        instead of modify the input programs inplace, this function will preserve the inputs and create new program for output.

        beside replace the serial op with its dist op, if user has defined other strategy in fleet.distributed_strategy, and if 
        those strategy need to transpile (modify) the forward network program, those forward program modification should also be done within this
        function in auto parallel scenario, in order to facilitate distributed inference/evaluation which need to DECOUPLE strategy specific forward transpilation with fleet.distributed_optimizer.minimize().

        by now the fleet.distributed_strategy that need transpile forward program are following: 
            1. AMP 
            2. Recompute 
            3. Pipeline 
            4. sharding

        Args:
            main_program (paddle.fluid.framework.program): serial main program with forward network only
            startup_program (paddle.fluid.framework.program): serial startup program with forward network only
        
        return:
            main_program (paddle.fluid.framework.program): distributed main program with forward network only
            startup_program (paddle.fluid.framework.program): distributed startup program with forward network only
        """

        dist_main_program, dist_startup_program = self.transpile_forward_impl(
            serial_main_program, serial_startup_program)
        return dist_main_program, dist_startup_program

    def apply_backward(self,
                       serial_loss,
                       serial_main_program,
                       serial_startup_program,
                       dist_main_program,
                       dist_startup_program,
                       parameter_list=None,
                       no_grad_set=None,
                       callbacks=None):
        """
        A complete training neural network is made up of forward and backward propagation. 
        This function is to generate the dist backward program for the distributed forward program.

        By now, the current automatical backward mechanism in paddle framework might NOT handle the backward generation for 
        some dist ops correctly, some so we now have two ways to genenate the backward program:
            1. dist_forward_program --> auto_backward --> dist_backward_program (if auto_backward could handle all dist op)
            2. serial_forward_program --> auto_backward --> serial_backward_program --> dist_op_backward_transpile --> dist_backward_program (if auto_backward could not handle all dist op)
        
        the backprogram is append the input dist program inplaced.

        Args:
            serial_loss (Variable) the loss in serial program that to be minimized 
            serial_main_program (paddle.fluid.framework.program): serial main program with forward network only
            serial_startup_program (paddle.fluid.framework.program): serial startup program with forward network only
            dist_main_program (paddle.fluid.framework.program): dist main program with forward network only
            dist_startup_program (paddle.fluid.framework.program): dist startup program with forward network only
            parameter_list (Iterable, optional): Iterable of ``Variable`` or ``Variable.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Variable``  or ``Variable.name`` that don't need
                to be updated. The default value is None.
            callbacks (list, optional): list of callable objects to run when appending backward
                operator for one parameter. The default value is None.
        
        return:
            params_grads (list) list of tuple that contain param and its grad variable
        """
        params_grads = self.apply_backward_impl(
            serial_loss, serial_main_program, serial_startup_program,
            dist_main_program, dist_startup_program)
        return params_grads

    def apply_optimize(self, user_define_optimizer, params_grads,
                       dist_main_program, dist_startup_program):
        """
        append update related ops to the program: clip, weight decay, ops
        filter optimize op if sharding is enable
        naive gradient synchronization before update

        Args:
            user_define_optimizer (paddle.fluid.optimizer): 
            params_grads (list) list of tuple that contain param and its grad variable
            dist_main_program (paddle.fluid.framework.program): dist main program with forward & backward network 
            dist_startup_program (paddle.fluid.framework.program): dist startup program with forward & backward  network 
        """

        optimize_ops = self.apply_optimize_impl(user_define_optimizer,
                                                params_grads, dist_main_program,
                                                dist_startup_program)

        return optimize_ops
