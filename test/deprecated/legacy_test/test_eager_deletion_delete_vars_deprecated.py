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

import os

import numpy as np

os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['CPU_NUM'] = '4'

import unittest
from functools import reduce

import paddle
from paddle import base

paddle.enable_static()

base.core._set_eager_deletion_mode(0.0, 1.0, True)


def simple_fc_net():
    image = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    hidden = image
    for _ in range(4):
        hidden = paddle.static.nn.fc(
            hidden,
            size=200,
            activation='tanh',
            bias_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            ),
        )
    prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3)
    optimizer.minimize(loss)
    return image, label, loss


def get_persistables_and_non_persistables(prog, fetch_list):
    num_block = prog.num_blocks
    persitables = set()
    non_persistables = set()
    for bid in range(num_block):
        block = prog.block(bid)
        for _, var in block.vars.items():
            if var.persistable or var.name in fetch_list:
                persitables.add(var.name)
            else:
                non_persistables.add(var.name)

    return persitables, non_persistables


class TestExecutor(unittest.TestCase):
    def test_executor_main(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for p in places:
            self.place = p
            with base.program_guard(base.Program(), base.Program()):
                with base.scope_guard(base.Scope()):
                    with base.unique_name.guard():
                        self.executor_main()

    def prepare_feed(self, image, label, dev_cnt=1):
        batch_size = 32 * dev_cnt
        image_shape = (batch_size, *image.shape[1:])
        label_shape = (batch_size, *label.shape[1:])

        image_np = np.random.random(size=image_shape).astype('float32')
        label_np = np.random.random_integers(
            low=0, high=9, size=label_shape
        ).astype('int64')

        return image_np, label_np

    def assertScopeVar(self, scope, persitables, non_persistables):
        outline_p_vars = []
        for name in persitables:
            var = scope.find_var(name)
            self.assertIsNotNone(var)
            t = var.get_tensor()
            if not t._is_initialized():
                outline_p_vars.append(name)

        outline_np_vars = []
        for name in non_persistables:
            var = scope.find_var(name)
            self.assertIsNotNone(var)
            t = var.get_tensor()
            if t._is_initialized():
                outline_np_vars.append(name)

        print(f'Non-alive persistable vars {outline_p_vars} in {persitables}')
        print(
            f'Alive non-persistable vars {outline_np_vars} in {non_persistables}'
        )
        self.assertEqual(len(outline_p_vars), 0)
        self.assertEqual(len(outline_np_vars), 0)

    def assert_gc_vars(self, program, skip_vars, non_persistable_vars):
        gc_vars = base.core._get_eager_deletion_vars(program.desc, skip_vars)
        self.assertEqual(len(gc_vars), program.num_blocks)
        gc_vars = reduce(lambda x, y: x + y, gc_vars[0])
        self.assertEqual(set(gc_vars), set(non_persistable_vars))

    def executor_main(self):
        image, label, loss = simple_fc_net()
        loss.persistable = False
        persistables, non_persistables = get_persistables_and_non_persistables(
            base.default_main_program(), [loss.name]
        )
        print(f'Non-persistable var number {len(non_persistables)}')
        print(non_persistables)

        self.assert_gc_vars(
            base.default_main_program(), [loss.name], non_persistables
        )

        exe = base.Executor(self.place)
        exe.run(base.default_startup_program())

        p = base.core.Place()
        p.set_place(self.place)
        exe = base.core.Executor(p)

        for _ in range(10):
            image_np, label_np = self.prepare_feed(image, label)
            base.global_scope().var(image.name).get_tensor().set(
                image_np, self.place
            )
            base.global_scope().var(label.name).get_tensor().set(
                label_np, self.place
            )
            # exe.run would not create local scope
            # so that we can detect whether gc clears temporary variables
            exe.run(
                base.default_main_program().desc,
                base.global_scope(),
                0,
                False,
                True,
                [loss.name],
            )
            self.assertScopeVar(
                base.global_scope(), persistables, non_persistables
            )


if __name__ == '__main__':
    unittest.main()
