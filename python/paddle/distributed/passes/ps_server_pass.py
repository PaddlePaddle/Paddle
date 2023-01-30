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

<<<<<<< HEAD
import logging

import paddle
from paddle.fluid.layers.learning_rate_scheduler import (
    exponential_decay,
    inverse_time_decay,
    natural_exp_decay,
    noam_decay,
)
from paddle.optimizer.lr import (
    ExponentialDecay,
    InverseTimeDecay,
    LRScheduler,
    NaturalExpDecay,
    NoamDecay,
)

from ..ps.utils.public import (
    get_optimize_ops,
    get_ps_endpoint,
    get_role_id,
    get_trainers,
)
from .pass_base import PassBase, register_pass
=======
import paddle
from ..ps.utils.public import *
from paddle.framework import core
from .pass_base import PassBase, register_pass
from paddle.optimizer.lr import LRScheduler
from paddle.optimizer.lr import ExponentialDecay, NoamDecay, PiecewiseDecay, NaturalExpDecay, InverseTimeDecay
from paddle.fluid.layers.learning_rate_scheduler import exponential_decay, noam_decay, piecewise_decay, natural_exp_decay, inverse_time_decay
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@register_pass("add_lr_decay_table_pass")
class AddLrDecayTablePass(PassBase):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(AddLrDecayTablePass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

<<<<<<< HEAD
    def _add_tensor_table(
        self,
        attrs,
        feed_var_name,
        fetch_var_name="",
        startup_program=None,
        main_program=None,
        tensor_table_class="",
    ):
=======
    def _add_tensor_table(self,
                          attrs,
                          feed_var_name,
                          fetch_var_name="",
                          startup_program=None,
                          main_program=None,
                          tensor_table_class=""):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        tensor_table_dict = {}
        tensor_table_dict[feed_var_name] = {}
        tensor_table_dict[feed_var_name]["feed_var_name"] = feed_var_name
        tensor_table_dict[feed_var_name]["fetch_var_name"] = fetch_var_name
        tensor_table_dict[feed_var_name]["startup_program"] = startup_program
        tensor_table_dict[feed_var_name]["main_program"] = main_program
        tensor_table_dict[feed_var_name][
<<<<<<< HEAD
            "tensor_table_class"
        ] = tensor_table_class
=======
            "tensor_table_class"] = tensor_table_class
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs['tensor_table'] = tensor_table_dict

    def _get_lr_sheduler_program(self, lr_sheduler, lr_decay_steps):
        schedler_decay = [
<<<<<<< HEAD
            'NoamDecay',
            'NaturalExpDecay',
            'InverseTimeDecay',
            'ExponentialDecay',
        ]

        decay_main_program = paddle.static.Program()
        decay_startup_program = paddle.static.Program()
        lr_name = ""

        if isinstance(lr_sheduler, ExponentialDecay):
            with paddle.static.program_guard(
                decay_main_program, decay_startup_program
            ):
                lr = exponential_decay(
                    1.0, lr_decay_steps, lr_sheduler.gamma, True
                )
=======
            'NoamDecay', 'NaturalExpDecay', 'InverseTimeDecay',
            'ExponentialDecay'
        ]

        decay_main_program = fluid.framework.Program()
        decay_startup_program = fluid.framework.Program()
        lr_name = ""

        if isinstance(lr_sheduler, ExponentialDecay):
            with fluid.program_guard(decay_main_program, decay_startup_program):
                lr = exponential_decay(1.0, lr_decay_steps, lr_sheduler.gamma,
                                       True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                lr_name = lr.name
                logging.warn(
                    "ExponentialDecay is set, staircase = True, global learning rate decay step is [ %d ], Change decay steps as follow: \n"
                    "\t strategy = paddle.distributed.fleet.DistributedStrategy() \n "
                    "\t strategy.a_sync = True \n"
                    "\t strategy.a_sync_configs= { 'lr_decay_steps' : YOUR_DECAY_STEP } \n"
<<<<<<< HEAD
                    % lr_decay_steps
                )
        elif isinstance(lr_sheduler, NoamDecay):
            with paddle.static.program_guard(
                decay_main_program, decay_startup_program
            ):
                lr = noam_decay(
                    lr_sheduler.d_model, lr_sheduler.warmup_steps, 1.0
                )
                lr_name = lr.name
                logging.warn(
                    "NoamDecay is set, warmup steps is [ %d ]"
                    % lr_sheduler.warmup_steps
                )
        elif isinstance(lr_sheduler, NaturalExpDecay):
            with paddle.static.program_guard(
                decay_main_program, decay_startup_program
            ):
                lr = natural_exp_decay(
                    1.0, lr_decay_steps, lr_sheduler.gamma, True
                )
=======
                    % lr_decay_steps)
        elif isinstance(lr_sheduler, NoamDecay):
            with fluid.program_guard(decay_main_program, decay_startup_program):
                lr = noam_decay(lr_sheduler.d_model, lr_sheduler.warmup_steps,
                                1.0)
                lr_name = lr.name
                logging.warn("NoamDecay is set, warmup steps is [ %d ]" %
                             lr_sheduler.warmup_steps)
        elif isinstance(lr_sheduler, NaturalExpDecay):
            with fluid.program_guard(decay_main_program, decay_startup_program):
                lr = natural_exp_decay(1.0, lr_decay_steps, lr_sheduler.gamma,
                                       True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                lr_name = lr.name
                logging.warn(
                    "NaturalExpDecay is set, staircase = True, global learning rate decay step is [ %d ], Change decay steps as follow: \n"
                    "\t strategy = paddle.distributed.fleet.DistributedStrategy() \n "
                    "\t strategy.a_sync = True \n"
                    "\t strategy.a_sync_configs= { 'lr_decay_steps' : YOUR_DECAY_STEP } \n"
<<<<<<< HEAD
                    % lr_decay_steps
                )
        elif isinstance(lr_sheduler, InverseTimeDecay):
            with paddle.static.program_guard(
                decay_main_program, decay_startup_program
            ):
                lr = inverse_time_decay(
                    1.0, lr_decay_steps, lr_sheduler.gamma, True
                )
=======
                    % lr_decay_steps)
        elif isinstance(lr_sheduler, InverseTimeDecay):
            with fluid.program_guard(decay_main_program, decay_startup_program):
                lr = inverse_time_decay(1.0, lr_decay_steps, lr_sheduler.gamma,
                                        True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                lr_name = lr.name
                logging.warn(
                    "InverseTimeDecay is set, staircase = True, global learning rate decay step is [ %d ], Change decay steps as follow: \n"
                    "\t strategy = paddle.distributed.fleet.DistributedStrategy() \n "
                    "\t strategy.a_sync = True \n"
                    "\t strategy.a_sync_configs= { 'lr_decay_steps' : YOUR_DECAY_STEP } \n"
<<<<<<< HEAD
                    % lr_decay_steps
                )
        else:
            raise ValueError(
                "Not supported current LearningRate strategy, please use follow decay strategy: {}".format(
                    schedler_decay
                )
            )
=======
                    % lr_decay_steps)
        else:
            raise ValueError(
                "Not supported current LearningRate strategy, please use follow decay strategy: {}"
                .format(schedler_decay))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return decay_main_program, decay_startup_program, lr_name

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
<<<<<<< HEAD
        if not hasattr(attrs['origin_main_program'], 'lr_sheduler'):
            return

        assert isinstance(
            attrs['origin_main_program'].lr_sheduler, LRScheduler
        ), "must be LRScheduler"

        ops = get_optimize_ops(attrs['origin_main_program'])
        (
            lr_decay_main_program,
            lr_decay_startup_program,
            lr_name,
        ) = self._get_lr_sheduler_program(
            attrs['origin_main_program'].lr_sheduler, attrs['lr_decay_steps']
        )
        self._add_tensor_table(
            attrs,
            "@LR_DECAY_COUNTER@",
            lr_name,
            lr_decay_startup_program,
            lr_decay_main_program,
            "GlobalStepTable",
        )
=======
        if hasattr(attrs['origin_main_program'], 'lr_sheduler') == False:
            return

        assert isinstance(attrs['origin_main_program'].lr_sheduler,
                          LRScheduler), "must be LRScheduler"

        ops = get_optimize_ops(attrs['origin_main_program'])
        lr_decay_main_program, lr_decay_startup_program, lr_name = self._get_lr_sheduler_program(
            attrs['origin_main_program'].lr_sheduler, attrs['lr_decay_steps'])
        self._add_tensor_table(attrs, "@LR_DECAY_COUNTER@", lr_name,
                               lr_decay_startup_program, lr_decay_main_program,
                               "GlobalStepTable")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return


@register_pass("add_listen_and_serv_pass")
class AddListenAndServPass(PassBase):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(AddListenAndServPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        opt = {
            "grad_to_block_id": None,
            "sparse_grad_to_param": None,
            "lr_decay_block_id": None,
            "dense_optimize_blocks": None,
            "sparse_optimize_blocks": None,
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            # runtime attribute
            "endpoint": get_ps_endpoint(attrs['role_maker']),
            "pserver_id": get_role_id(attrs['role_maker']),
            "Fanin": get_trainers(attrs['role_maker']),
            "distributed_mode": attrs['ps_mode'],
            "rpc_get_thread_num": -1,
            "rpc_send_thread_num": -1,
<<<<<<< HEAD
            "rpc_prefetch_thread_num": -1,
        }
        main_program.global_block().append_op(
            type="listen_and_serv", inputs={'X': []}, outputs={}, attrs=opt
        )
=======
            "rpc_prefetch_thread_num": -1
        }
        main_program.global_block().append_op(type="listen_and_serv",
                                              inputs={'X': []},
                                              outputs={},
                                              attrs=opt)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@register_pass("add_rpc_global_flags_pass")
class AddRpcGlobalFlagsPass(PassBase):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(AddRpcGlobalFlagsPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        pass


@register_pass("add_optimizer_pass")
class AddOptimizerPass(PassBase):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(AddOptimizerPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        pass


@register_pass("add_geo_optimizer_pass")
class AddGeoOptimizerPass(PassBase):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(AddGeoOptimizerPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        pass


@register_pass("build_pserver_startup_program_pass")
class BuildPserverStartupProgramPass(PassBase):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(BuildPserverStartupProgramPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        pass


@register_pass("delete_unused_in_startup_pass")
class DeleteUnusedInStartupPass(PassBase):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(DeleteUnusedInStartupPass, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        pass
