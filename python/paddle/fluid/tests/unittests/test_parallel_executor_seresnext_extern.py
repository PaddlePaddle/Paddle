# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from functools import partial

import paddle.fluid as fluid
fluid.core._set_fuse_parameter_group_size(3)
fluid.core._set_fuse_parameter_memory_size(131072)

import paddle.fluid.core as core
from parallel_executor_test_base import TestParallelExecutorBase
from simple_nets import init_data
import unittest
from seresnext_base_model import model, _feed_dict, _iter, _batch_size, optimizer, img_shape

gpu_img, gpu_label = init_data(
    batch_size=_batch_size(), img_shape=img_shape, label_range=999)
cpu_img, cpu_label = init_data(
    batch_size=_batch_size(), img_shape=img_shape, label_range=999)
feed_dict_gpu = {"image": gpu_img, "label": gpu_label}
feed_dict_cpu = {"image": cpu_img, "label": cpu_label}


def _get_result_of_origin_model(use_cuda):
    global remove_bn
    global remove_dropout
    remove_bn = True
    remove_dropout = True
    first_loss, last_loss = TestParallelExecutorBase.check_network_convergence(
        model,
        feed_dict=_feed_dict(use_cuda),
        iter=_iter(use_cuda),
        batch_size=_batch_size(),
        use_cuda=use_cuda,
        use_reduce=False,
        optimizer=optimizer)

    return first_loss, last_loss


origin_cpu_first_loss, origin_cpu_last_loss = _get_result_of_origin_model(False)
if core.is_compiled_with_cuda():
    origin_gpu_first_loss, origin_gpu_last_loss = _get_result_of_origin_model(
        True)


def _get_origin_result(use_cuda):
    if use_cuda:
        assert core.is_compiled_with_cuda(), "Doesn't compiled with CUDA."
        return origin_gpu_first_loss, origin_gpu_last_loss
    return origin_cpu_first_loss, origin_cpu_last_loss


class TestResnet(TestParallelExecutorBase):
    def _compare_result_with_origin_model(self,
                                          get_origin_result,
                                          check_func_2,
                                          use_cuda,
                                          delta2=1e-5,
                                          compare_seperately=True,
                                          rm_drop_out=False,
                                          rm_bn=False):
        if use_cuda and not core.is_compiled_with_cuda():
            return

        func_1_first_loss, func_1_last_loss = get_origin_result(use_cuda)
        func_2_first_loss, func_2_last_loss = check_func_2(
            model,
            feed_dict=_feed_dict(use_cuda),
            iter=_iter(use_cuda),
            batch_size=_batch_size(),
            use_cuda=use_cuda)

        if compare_seperately:
            for loss in zip(func_1_first_loss, func_2_first_loss):
                self.assertAlmostEquals(loss[0], loss[1], delta=1e-5)
            for loss in zip(func_1_last_loss, func_2_last_loss):
                self.assertAlmostEquals(loss[0], loss[1], delta=delta2)
        else:
            self.assertAlmostEquals(
                np.mean(func_1_first_loss), func_2_first_loss[0], delta=1e-5)
            self.assertAlmostEquals(
                np.mean(func_1_last_loss), func_2_last_loss[0], delta=delta2)

    def test_seresnext_with_learning_rate_decay(self):
        # NOTE(zcd): This test is compare the result of use parallel_executor and executor,
        # and the result of drop_out op and batch_norm op in this two executor
        # have diff, so the two ops should be removed from the model.
        check_func_1 = _get_origin_result
        check_func_2 = partial(
            self.check_network_convergence,
            optimizer=optimizer,
            use_parallel_executor=False)
        self._compare_result_with_origin_model(
            check_func_1,
            check_func_2,
            use_cuda=False,
            rm_drop_out=True,
            rm_bn=True,
            compare_seperately=False,
            delta2=1e-3)
        self._compare_result_with_origin_model(
            check_func_1,
            check_func_2,
            use_cuda=True,
            rm_drop_out=True,
            rm_bn=True,
            compare_seperately=False)

    def test_seresnext_with_fused_all_reduce(self):
        # NOTE(zcd): In order to make the program faster,
        # this unit test remove drop_out and batch_norm.
        check_func_1 = _get_origin_result
        check_func_2 = partial(
            self.check_network_convergence,
            optimizer=optimizer,
            fuse_all_reduce_ops=True)
        self._compare_result_with_origin_model(
            check_func_1,
            check_func_2,
            use_cuda=False,
            rm_drop_out=True,
            rm_bn=True)
        self._compare_result_with_origin_model(
            check_func_1,
            check_func_2,
            use_cuda=True,
            rm_drop_out=True,
            rm_bn=True,
            delta2=1e-2)


if __name__ == '__main__':
    unittest.main()
