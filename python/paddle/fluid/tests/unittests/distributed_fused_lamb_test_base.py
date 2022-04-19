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

import argparse
import os
import paddle
import paddle.fluid.core as core
import paddle.distributed.fleet as fleet
from paddle.incubate import DistributedFusedLamb
from paddle.vision.models import resnet18 as resnet
from paddle.distributed.fleet.meta_optimizers.common import CollectiveHelper
from paddle.fluid.clip import ClipGradBase
import paddle.nn as nn
import numpy as np
import os
import unittest
from paddle.distributed.fleet.meta_optimizers.common import is_optimizer_op, is_backward_op
from paddle.fluid.clip import _clip_by_global_norm_using_mp_type
import distutils


def get_role_maker():
    return fleet.PaddleCloudRoleMaker(is_collective=True)


def set_seed(seed):
    paddle.seed(seed)
    rank = paddle.distributed.get_rank()
    np_seed = seed + rank
    np.random.seed(np_seed)


def set_gradient_persistable(program):
    block = program.global_block()
    params = []
    grads = []
    for p in block.all_parameters():
        p_name = p.name
        g_name = p_name + '@GRAD'
        g = block.vars.get(g_name)
        if g is None:
            continue
        g.persistable = True
        params.append(p)
        grads.append(g)
    return params, grads


def prune_fwd_bwd_ops(program, start_idx):
    for i in reversed(range(start_idx)):
        program.global_block()._remove_op(i, sync=False)
    program._sync_with_cpp()

    ops = program.global_block().ops
    all_vars = set(program.global_block().vars.keys())
    for op in ops:
        args = op.input_arg_names + op.output_arg_names
        for arg in args:
            if arg in all_vars:
                all_vars.remove(arg)

    for var in all_vars:
        program.global_block()._remove_var(var)
    program._sync_with_cpp()


class GradClipDecorator(ClipGradBase):
    def __init__(self, clip, clip_after_allreduce):
        self.clip = clip
        self.clip_after_allreduce = clip_after_allreduce

    def _dygraph_clip(self, params_grads):
        raise NotImplementedError()

    def _insert_allreduce_ops(self, params_grads):
        world_size = paddle.distributed.get_world_size()
        if world_size == 1:
            return
        block = params_grads[0][0].block
        scale = 1.0 / world_size
        # scale = 1.0
        for p, g in params_grads:
            block.append_op(
                type='c_allreduce_sum',
                inputs={'X': [g]},
                outputs={'Out': [g]},
                attrs={'ring_id': 0,
                       'use_calc_stream': True})
            block.append_op(
                type='scale',
                inputs={'X': [g]},
                outputs={'Out': [g]},
                attrs={'scale': scale})

    def _static_clip(self, params_grads):
        if self.clip_after_allreduce:
            self._insert_allreduce_ops(params_grads)

        params_grads = self.clip(params_grads)
        if not self.clip_after_allreduce:
            self._insert_allreduce_ops(params_grads)
        return params_grads


class IdentityGradClip(ClipGradBase):
    def _dygraph_clip(self, params_grads):
        return params_grads

    def _static_clip(self, params_grads):
        return params_grads


def run_model(use_distributed_lamb, use_fp16, use_master_param_norm, **kwargs):
    nranks = paddle.distributed.get_world_size()

    set_seed(1000)
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        with paddle.fluid.unique_name.guard():
            with paddle.static.amp.fp16_guard():
                image = paddle.static.data(
                    name='image',
                    shape=[None, 3, 224, 224],
                    dtype=paddle.float32)
                label = paddle.static.data(
                    name='label', shape=[None, 1], dtype=paddle.int64)
                model = resnet()
                pred = model(image)
                loss_fn = paddle.nn.loss.CrossEntropyLoss()
                loss = loss_fn(pred, label)

            grad_clip = kwargs.get('grad_clip', None)
            clip_after_allreduce = kwargs.get('clip_after_allreduce', True)

            parameters = [p.name for p in main.all_parameters()]
            exclude_fn = lambda var: var.name in parameters[::4]
            kwargs['exclude_from_weight_decay_fn'] = exclude_fn
            kwargs['lamb_weight_decay'] = 0.1

            if use_distributed_lamb:
                optimizer_class = DistributedFusedLamb
                kwargs = dict(kwargs)
                kwargs['is_grad_scaled_by_nranks'] = False
                kwargs['use_master_param_norm'] = use_master_param_norm
            else:
                optimizer_class = paddle.optimizer.Lamb
                kwargs = dict(kwargs)
                kwargs.pop('clip_after_allreduce', None)
                kwargs.pop('alignment', None)
                base_clip = grad_clip if grad_clip is not None else IdentityGradClip(
                )
                kwargs['grad_clip'] = GradClipDecorator(base_clip,
                                                        clip_after_allreduce)

            optimizer = optimizer_class(**kwargs)
            get_parameter = optimizer._get_parameter
            amp_list = paddle.static.amp.AutoMixedPrecisionLists(
                custom_white_list=[
                    'batch_norm', 'batch_norm_grad', 'conv2d', 'conv2d_grad'
                ])
            if use_fp16:
                if not use_distributed_lamb:
                    optimizer._multi_precision = True
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    amp_list,
                    init_loss_scaling=1.0,
                    use_dynamic_loss_scaling=False,
                    use_pure_fp16=use_fp16,
                    use_fp16_guard=use_fp16)

            params_grads = optimizer.backward(loss, startup)
            op_num = len(main.global_block().ops)
            if use_fp16:
                optimizer.apply_optimize(loss, startup, params_grads)
            else:
                optimizer.apply_gradients(params_grads)

        if nranks > 1:
            collective_helper = CollectiveHelper(role_maker=get_role_maker())
            collective_helper.update_startup_program(startup)
        set_gradient_persistable(startup)
        params, grads = set_gradient_persistable(main)
        prune_fwd_bwd_ops(main, op_num)

    def pd_dtype_to_np_dtype(pd_dtype):
        if pd_dtype == paddle.float32:
            return np.float32
        elif pd_dtype == paddle.float16:
            return np.float16
        else:
            raise ValueError("supported dtype {}".format(pd_dtype))

    def gen_random_grad_tensor(grad):
        np_dtype = pd_dtype_to_np_dtype(grad.dtype)
        grad_np = np.random.random(size=grad.shape).astype(np_dtype)
        grad_t = core.Tensor()
        grad_t.set(grad_np, paddle.CPUPlace())
        return grad_t

    def reader():
        for _ in range(5):
            yield dict(
                [(grad.name, gen_random_grad_tensor(grad)) for grad in grads])

    scope = paddle.static.Scope()
    fetch_list = params
    fetches = None
    with paddle.static.scope_guard(scope):
        dev_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = paddle.CUDAPlace(dev_id)
        exe = paddle.static.Executor(place)
        exe.run(startup)
        if use_fp16:
            optimizer.amp_init(place)

        master_p_ts = []
        for p in params:
            p_ts = get_parameter(p.name)
            assert len(p_ts) == 2
            if p_ts[1] is not None:
                master_p_ts.append(p_ts[1])
        if use_fp16:
            assert len(master_p_ts) > 0
        else:
            assert len(master_p_ts) == 0

        for feed in reader():
            fetches = exe.run(main, feed=feed, fetch_list=fetch_list)
    return fetches


class TestDistributedFusedLamb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            return

        paddle.enable_static()
        paddle.set_flags({'FLAGS_cudnn_deterministic': True})
        _clip_by_global_norm_using_mp_type(True)
        fleet.init(role_maker=get_role_maker())

    def config(self):
        clip_after_allreduce = bool(
            distutils.util.strtobool(
                os.getenv('CLIP_AFTER_ALLREDUCE', 'True')))
        max_global_norm = float(os.getenv('MAX_GLOBAL_NORM', -1.0))
        print('clip_after_allreduce = {}, max_global_norm = {}'.format(
            clip_after_allreduce, max_global_norm))
        return {
            'clip_after_allreduce': clip_after_allreduce,
            'grad_clip': paddle.nn.ClipGradByGlobalNorm(max_global_norm)
            if max_global_norm > 0 else None,
        }

    def run_main(self, use_fp16, use_master_param_norm=True):
        if not paddle.is_compiled_with_cuda():
            return

        if not use_fp16:
            self.assertTrue(use_master_param_norm)

        base_config = self.config()
        config1 = dict(base_config)
        config1['use_distributed_lamb'] = True
        config1['use_fp16'] = use_fp16
        config1['use_master_param_norm'] = use_master_param_norm

        config2 = dict(base_config)
        config2['use_distributed_lamb'] = False
        config2['use_fp16'] = use_fp16
        config2['use_master_param_norm'] = use_master_param_norm

        result1 = run_model(**config1)
        result2 = run_model(**config2)
        self.assertEqual(len(result1), len(result2))

        if use_fp16:
            atol = 8e-4 if use_master_param_norm else 1e-3
        else:
            atol = 1e-7
        for ret1, ret2 in zip(result1, result2):
            max_diff = np.max(np.abs(ret1 - ret2))
            msg = 'max_diff = {} atol = {} when use_fp16 = {} , use_master_param_norm = {}'.format(
                max_diff, atol, use_fp16, use_master_param_norm)
            self.assertTrue(max_diff < atol, msg)
            print(msg)

    def test_main(self):
        self.run_main(use_fp16=False)
        self.run_main(use_fp16=True, use_master_param_norm=True)
        self.run_main(use_fp16=True, use_master_param_norm=False)

        touch_file_name = os.environ.get('SUCCESS_TOUCH_FILE')
        if touch_file_name:
            with open(touch_file_name, 'w') as f:
                f.write('success')


if __name__ == "__main__":
    unittest.main()
