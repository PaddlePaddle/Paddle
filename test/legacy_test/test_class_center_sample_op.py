#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import OpTest, paddle_static_guard

import paddle
from paddle.base import core


def class_center_sample_numpy(label, classes_list, num_samples):
    unique_label = np.unique(label)
    nranks = len(classes_list)
    class_interval = np.cumsum(np.insert(classes_list, 0, 0))
    pos_class_center_per_device = []
    unique_label_per_device = []

    for i in range(nranks):
        index = np.logical_and(
            unique_label >= class_interval[i],
            unique_label < class_interval[i + 1],
        )
        pos_class_center_per_device.append(
            unique_label[index] - class_interval[i]
        )
        unique_label_per_device.append(unique_label[index])

    num_samples_per_device = []
    for pos_class_center in pos_class_center_per_device:
        num_samples_per_device.append(max(len(pos_class_center), num_samples))
    sampled_class_interval = np.cumsum(np.insert(num_samples_per_device, 0, 0))

    remapped_dict = {}
    for i in range(nranks):
        for idx, v in enumerate(
            unique_label_per_device[i], sampled_class_interval[i]
        ):
            remapped_dict[v] = idx

    remapped_label = []
    for l in label:
        remapped_label.append(remapped_dict[l])

    return np.array(remapped_label), np.array(pos_class_center_per_device)


def python_api(
    label,
    num_classes=1,
    num_samples=1,
    ring_id=0,
    rank=0,
    nranks=0,
    fix_seed=False,
    seed=0,
):
    return paddle.nn.functional.class_center_sample(
        label, num_classes=num_classes, num_samples=num_samples, group=None
    )


class TestClassCenterSampleOp(OpTest):
    def initParams(self):
        self.op_type = "class_center_sample"
        self.python_api = python_api
        self.batch_size = 20
        self.num_samples = 6
        self.num_classes = 10
        self.seed = 2021

    def init_dtype(self):
        self.dtype = np.int64

    def init_fix_seed(self):
        self.fix_seed = True

    def setUp(self):
        self.initParams()
        self.init_dtype()
        self.init_fix_seed()
        label = np.random.randint(
            0, self.num_classes, (self.batch_size,), dtype=self.dtype
        )

        remapped_label, sampled_class_center = class_center_sample_numpy(
            label, [self.num_classes], self.num_samples
        )

        self.inputs = {'Label': label}
        self.outputs = {
            'RemappedLabel': remapped_label.astype(self.dtype),
            'SampledLocalClassCenter': sampled_class_center.astype(self.dtype),
        }

        self.attrs = {
            'num_classes': self.num_classes,
            'num_samples': self.num_samples,
            'seed': self.seed,
            'fix_seed': self.fix_seed,
        }

    def test_check_output(self):
        self.check_output(
            no_check_set=['SampledLocalClassCenter'],
            check_pir=True,
            check_symbol_infer=False,
        )


class TestClassCenterSampleOpINT32(TestClassCenterSampleOp):
    def init_dtype(self):
        self.dtype = np.int32


class TestClassCenterSampleOpFixSeed(TestClassCenterSampleOp):
    def init_fix_seed(self):
        self.fix_seed = True


class TestClassCenterSampleV2(unittest.TestCase):
    def setUp(self):
        self.initParams()
        np.random.seed(self.seed)
        paddle.framework.random._manual_program_seed(2021)
        self.places = [paddle.base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.base.CUDAPlace(0))

    def initParams(self):
        self.batch_size = 10
        self.num_samples = 6
        self.num_classes = 20
        self.seed = 0
        self.init_dtype()

    def init_dtype(self):
        self.dtype = np.int64

    def test_static(self):
        with paddle_static_guard():
            for place in self.places:
                self.check_static_result(place=place)

    def check_static_result(self, place):
        with paddle_static_guard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                label_np = np.random.randint(
                    0, self.num_classes, (self.batch_size,), dtype=self.dtype
                )

                label = paddle.static.data(
                    name='label', shape=[self.batch_size], dtype=self.dtype
                )
                (
                    remapped_label,
                    sampled_class_index,
                ) = paddle.nn.functional.class_center_sample(
                    label, self.num_classes, self.num_samples
                )

                (
                    remapped_label_np,
                    sampled_class_center_np,
                ) = class_center_sample_numpy(
                    label_np, [self.num_classes], self.num_samples
                )
                exe = paddle.base.Executor(place)
                [remapped_label_res, sampled_class_index_res] = exe.run(
                    feed={'label': label_np},
                    fetch_list=[remapped_label, sampled_class_index],
                )
                np.testing.assert_allclose(
                    remapped_label_res, remapped_label_np
                )
                np.testing.assert_allclose(
                    sampled_class_index_res[: len(sampled_class_center_np[0])],
                    sampled_class_center_np[0],
                )

    def test_dynamic(self):
        for place in self.places:
            self.check_dynamic_result(place=place)

    def check_dynamic_result(self, place):
        with paddle.base.dygraph.guard(place):
            label_np = np.random.randint(
                0, self.num_classes, (self.batch_size,), dtype=self.dtype
            )
            label = paddle.to_tensor(label_np, dtype=self.dtype)

            (
                remapped_label,
                sampled_class_index,
            ) = paddle.nn.functional.class_center_sample(
                label, self.num_classes, self.num_samples
            )

            (
                remapped_label_np,
                sampled_class_center_np,
            ) = class_center_sample_numpy(
                label_np, [self.num_classes], self.num_samples
            )

            remapped_label_res = remapped_label.numpy()
            sampled_class_index_res = sampled_class_index.numpy()
            np.testing.assert_allclose(remapped_label_res, remapped_label_np)
            np.testing.assert_allclose(
                sampled_class_index_res[: len(sampled_class_center_np[0])],
                sampled_class_center_np[0],
            )


class TestClassCenterSampleV2INT32(TestClassCenterSampleV2):
    def init_dtype(self):
        self.dtype = np.int32


class TestClassCenterSampleAPIError(unittest.TestCase):
    def setUp(self):
        self.initParams()
        np.random.seed(self.seed)
        self.places = [paddle.base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.base.CUDAPlace(0))

    def initParams(self):
        self.batch_size = 20
        self.num_samples = 15
        self.num_classes = 10
        self.seed = 2021
        self.init_dtype()

    def init_dtype(self):
        self.dtype = np.int64

    def test_dynamic_errors(self):
        def test_num_samples():
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    label_np = np.random.randint(
                        0,
                        self.num_classes,
                        (self.batch_size,),
                        dtype=self.dtype,
                    )
                    label = paddle.to_tensor(label_np)

                    (
                        remapped_label,
                        sampled_class_index,
                    ) = paddle.nn.functional.class_center_sample(
                        label, self.num_classes, self.num_samples
                    )

        self.assertRaises(ValueError, test_num_samples)


class TestClassCenterSampleAPIError1(unittest.TestCase):
    def setUp(self):
        self.initParams()
        np.random.seed(self.seed)
        self.places = [paddle.base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.base.CUDAPlace(0))

    def initParams(self):
        self.batch_size = 5
        self.num_samples = 5
        self.num_classes = 10
        self.seed = 2021
        self.init_dtype()

    def init_dtype(self):
        self.dtype = np.int64

    def test_dynamic_errors(self):
        def test_empty_label():
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    label = paddle.to_tensor(np.array([], dtype=self.dtype))

                    (
                        remapped_label,
                        sampled_class_index,
                    ) = paddle.nn.functional.class_center_sample(
                        label, self.num_classes, self.num_samples
                    )

        def test_group_value():
            for place in self.places:
                with paddle.base.dygraph.guard(place):
                    label_np = np.random.randint(
                        0,
                        self.num_classes,
                        (self.batch_size,),
                        dtype=self.dtype,
                    )
                    label = paddle.to_tensor(label_np)

                    (
                        remapped_label,
                        sampled_class_index,
                    ) = paddle.nn.functional.class_center_sample(
                        label, self.num_classes, self.num_samples, group=True
                    )

        self.assertRaises(ValueError, test_empty_label)
        self.assertRaises(ValueError, test_group_value)


if __name__ == '__main__':
    unittest.main()
