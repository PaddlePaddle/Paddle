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
# limitations under the License.

######

from functools import reduce

import paddle
from paddle import framework
from paddle.fluid.dygraph import base as imperative_base

from ...utils.log_util import logger


def _is_trainable(param):
    return not param.stop_gradient


class DygraphShardingOptimizer:
    """
    A wrapper for Sharding Optimizer in Dygraph.

    .. warning: DygraphShardingOptimizer is experimental and subject to change.

    .. ZeRO: https://arxiv.org/abs/1910.02054

    """

    # TODO (JZ-LIANG)
    # TO support following featrues in future:
    # 1. fused update parameter sync
    # 2. parameters_groups
    # 3. dynamic trainable params, which is the case bewteen pretraining and finetuning
    # 4. option to choose fuse comm (more GPU MEM need) or un-fuse comm

    def __init__(self, optimizer, hcg):
        # TODO(pangengzheng): support param_groups
        if isinstance(optimizer._parameter_list[0], dict):
            raise TypeError(
                "Do not support param_groups now, please set optimizer._parameter_list as a list of Parameter"
            )
        if not hasattr(optimizer, '_apply_optimize') or not callable(
            optimizer._apply_optimize
        ):
            raise ValueError(
                "the optimzier object should have _apply_optimize function"
            )

        # the self._parameter_list holds the whole model paramters
        self._parameter_list = optimizer._parameter_list
        self._inner_opt = optimizer
        self._hcg = hcg
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        self._rank2params = self._partition_parameters()
        self._param2rank = self._map_param_to_rank()

        # update parameter_list of the inner_opt
        # Note: The inner_opt should promise that it only use its self._parameter_list.
        # Recursively use the self._inner_opt._inner_opt._parameter_list is not allowed.
        print("inner_opt:", self._inner_opt)
        if hasattr(self._inner_opt, '_inner_opt'):
            print(
                "self._inner_opt params len:{}, self._inner_opt._inner_opt params len:{}".format(
                    len(self._inner_opt._parameter_list),
                    len(self._inner_opt._inner_opt._parameter_list),
                )
            )
        print(f"self._parameter_list len:{len(self._parameter_list)}")
        inner_opt = self._inner_opt
        while hasattr(inner_opt, '_parameter_list'):
            inner_opt._parameter_list = self._rank2params[self._sharding_rank]
            if hasattr(inner_opt, '_inner_opt'):
                inner_opt = inner_opt._inner_opt
            else:
                break
        #
        # self._set_inner_opt_attr('_parameter_list', self._rank2params[self._sharding_rank])
        print(
            "after parameters update, params len:%d"
            % (len(self._inner_opt._parameter_list))
        )
        print(f"self._parameter_list len:{len(self._parameter_list)}")

    def clear_grad(self, set_to_zero=True):
        """
        should clear grad for all parameters in model
        """
        #
        for p in self._parameter_list:
            if hasattr(p, "main_grad") and p.main_grad is not None:
                assert p._grad_ivar() is None
                if set_to_zero:
                    p.main_grad.zero_()
                else:
                    p.main_grad._clear()
                    p.main_grad = None
            elif not hasattr(p, "main_grad"):
                p.clear_gradient(set_to_zero)

    def _partition_parameters(self):
        """
        Partitions parameters among sharding ranks.

        Return:
        Dict[int, List]
        """
        # TODO(JZ-LIANG) support multiple partition methods
        # method1: greedy even but unorder
        # method2: roughly even with oreder

        mapping = {}
        for rank_ in range(self._sharding_world_size):
            mapping[rank_] = []
        sizes = [0] * self._sharding_world_size
        for param in self._parameter_list:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param.shape)
            assert (
                numel > 0
            ), "param [{}] should larger than 0, but it is [{}]".format(
                param.name, numel
            )
            sizes[rank] += numel

        return mapping

    def _map_param_to_rank(self):
        """
        mapping parameters to the shard which holds it.

        Return:
        Dict[str, int]
        """
        mapping = {}
        for rank, params in self._rank2params.items():
            for param in params:
                mapping[param.name] = rank
        return mapping

    @staticmethod
    def reduce_gradients(parameter_list, hcg):
        # TODO allreduce --> reduce
        # TODO merge grad / nrank with dp
        logger.debug("sharding start gradients sync")
        with framework.no_grad():
            sharding_nrank = hcg.get_sharding_parallel_group().nranks
            for param in parameter_list:
                g_var = None
                if param.trainable and (param._grad_ivar() is not None):
                    g_var = param._grad_ivar()
                if param.trainable and hasattr(param, "main_grad"):
                    assert (
                        param._grad_ivar() is None
                    ), "param.grad should be None when using main_grad"
                    g_var = param.main_grad
                if g_var is not None:
                    g_var.scale_(1.0 / sharding_nrank)
                    paddle.distributed.all_reduce(
                        g_var,
                        group=hcg.get_sharding_parallel_group(),
                        sync_op=True,
                    )

    def _sharding_sync_parameters(self):
        """
        sync parameter across sharding group
        """
        # TODO speed up this functional

        print("sharding start sync parameters")
        with framework.no_grad():
            # TODO detach not need (?)
            for rank, params in self._rank2params.items():
                for param in params:
                    paddle.distributed.broadcast(
                        param,
                        # the collective API need src rank to be the global rank id
                        # instead of the relative logic rank id within group
                        src=self._hcg.get_sharding_parallel_group().ranks[rank],
                        group=self._hcg.get_sharding_parallel_group(),
                        sync_op=True,
                    )

    def _update_trainable(self):
        """
        allow user to update trainable parameters list during training
        """
        raise NotImplementedError

    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        # NOTE in dygraph mode, the only different between step and minimize is that minimize
        # allow user to customize the parameters for updating on each step

        input_param_names = {param.name for param in parameters}
        parameters = list(
            filter(
                lambda x: x.name in input_param_names,
                self._rank2params[self._sharding_rank],
            )
        )
        result = self._inner_opt.minimize(
            loss, startup_program, parameters, no_grad_set
        )

        # sync parameters across sharding ranks
        self._sharding_sync_parameters()

        return result

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly

        # # hack to grad_clip all parameters,
        # # otherwise the self._inner_opt will only grad_clip the self._rank2params[self._sharding_rank] params
        # origin_clip = self._inner_opt._grad_clip

        # def set_inner_opt_grad_clip(grad_clip_value):
        #     inner_opt = self._inner_opt
        #     while hasattr(inner_opt, '_grad_clip'):
        #         inner_opt._grad_clip = grad_clip_value
        #         if (
        #             hasattr(inner_opt, '_inner_opt')
        #             and inner_opt._inner_opt is not None
        #         ):
        #             inner_opt = inner_opt._inner_opt
        #         else:
        #             break

        # if not isinstance(self._parameter_list[0], dict):
        #     params_grads = []
        #     for param in self._parameter_list:
        #         if (
        #             hasattr(param, "regularizer")
        #             and param.regularizer is not None
        #         ):
        #             raise ValueError(
        #                 "param {} should not has the regularizer attribute".format(
        #                     param.name
        #                 )
        #             )
        #         if param.stop_gradient:
        #             continue
        #         grad_var = param._grad_ivar()
        #         if hasattr(param, "main_grad") and param.main_grad is not None:
        #             grad_var = param.main_grad
        #         params_grads.append((param, grad_var))
        #     print(
        #         "dygraph_sharding inner_opt:{}, grad_clip:{}".format(
        #             self._inner_opt, self._inner_opt._grad_clip
        #         )
        #     )
        #     print(f"clip grad params_grads len:{len(params_grads)}")
        #     if hasattr(self._inner_opt._grad_clip, 'not_sharding_stage1'):
        #         self._inner_opt._grad_clip.not_sharding_stage1 = False
        #     # def set_inner_opt_grad_clip_attr(attr_name, value):
        #     #     inner_opt = self._inner_opt
        #     #     while hasattr(inner_opt._grad_clip, attr_name):
        #     #         setattr(inner_opt._grad_clip, attr_name, value)
        #     #         if hasattr(inner_opt, '_inner_opt'):
        #     #             inner_opt = inner_opt._inner_opt
        #     #         else:
        #     #             break
        #     # set_inner_opt_grad_clip_attr('not_sharding_stage1', False)
        #     params_grads = self._inner_opt._grad_clip(params_grads)
        #     print(f"after grad clip, params_grads len:{len(params_grads)}")
        #     # set inner_opt._grad_clip None to avoid repeatedly grad_clip gradients inside inner_opt._apply_optimize
        #     set_inner_opt_grad_clip(None)
        #     # self._set_inner_opt_attr('_grad_clip', None)
        #     update_param_names = [
        #         p.name for p in self._rank2params[self._sharding_rank]
        #     ]
        #     update_params_grads = [
        #         (p, g) for p, g in params_grads if p.name in update_param_names
        #     ]
        #     print(
        #         "update_params_grads len:{}, non update params len:{}".format(
        #             len(update_params_grads), len(params_grads)
        #         )
        #     )

        # self._inner_opt.step() might be ok? because we already update the self._inner_opt._parameter_list
        self._inner_opt.step()
        # update_params_grads = []
        # for param in self._rank2params[self._sharding_rank]:
        #     if param.stop_gradient:
        #         continue
        #     grad_var = param._grad_ivar()
        #     if hasattr(param, "main_grad") and param.main_grad is not None:
        #         grad_var = param.main_grad
        #     update_params_grads.append((param, grad_var))
        # #
        # self._apply_optimize(
        #     loss=None,
        #     startup_program=None,
        #     params_grads=update_params_grads,
        # )
        # print(
        #     "after clip, dygraph_sharding inner_opt:{}, grad_clip:{}".format(
        #         self._inner_opt, self._inner_opt._grad_clip
        #     )
        # )
        # # restore the grad clip
        # set_inner_opt_grad_clip(origin_clip)
        # # self._set_inner_opt_attr('_grad_clip', origin_clip)

        # sync parameters across sharding ranks
        self._sharding_sync_parameters()

    def _set_inner_opt_attr(self, attr_name, value):
        inner_opt = self._inner_opt
        inner_opt_name = '_inner_opt'
        if not isinstance(attr_name, str):
            raise TypeError(
                "attr_name should be str type, but is {}".format(
                    type(attr_name)
                )
            )
        while hasattr(inner_opt, attr_name):
            setattr(inner_opt, attr_name, value)
            if (
                hasattr(inner_opt, inner_opt_name)
                and getattr(inner_opt, inner_opt_name, None) is not None
            ):
                inner_opt = getattr(inner_opt, inner_opt_name, None)
            else:
                break

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
