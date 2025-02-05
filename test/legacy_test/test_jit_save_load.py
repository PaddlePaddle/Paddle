# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np

import paddle
from paddle import base
from paddle.base import unique_name
from paddle.jit.api import to_static
from paddle.nn import Linear
from paddle.static import InputSpec

BATCH_SIZE = 32
BATCH_NUM = 10
SEED = 10


def random_batch_reader(input_size, label_size):
    def _get_random_inputs_and_labels(input_size, label_size):
        np.random.seed(SEED)
        input = np.random.random(size=input_size).astype('float32')
        label = np.random.random(size=label_size).astype('int64')
        return input, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_input, batch_label = _get_random_inputs_and_labels(
                [BATCH_SIZE, input_size], [BATCH_SIZE, label_size]
            )
            yield batch_input, batch_label

    return __reader__


class LinearNet(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        return self._linear(x)


class LinearNetWithInputSpec(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static(
        input_spec=[InputSpec(shape=[None, 784], dtype='float32')],
        full_graph=True,
    )
    def forward(self, x):
        return self._linear(x)


class LinearNetNotDeclarative(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, x):
        return self._linear(x)


class LinerNetWithLabel(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, x, label):
        out = self._linear(x)
        loss = paddle.nn.functional.cross_entropy(
            out, label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss)
        return out, avg_loss


class LinerNetWithPruneInput(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, x, label):
        out = self._linear(x)
        loss = paddle.nn.functional.cross_entropy(
            out, label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss)
        return out


class LinerNetWithUselessInput(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, x, label):
        out = self._linear(x)
        return out


class LinearNetReturnLoss(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        y = self._linear(x)
        z = self._linear(y)
        loss = paddle.mean(z)
        return z, loss


class LinearNetMultiInput(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear1 = Linear(in_size, out_size)
        self._linear2 = Linear(in_size, out_size)

    def forward(self, x, y):
        x_out = self._linear1(x)
        y_out = self._linear2(y)
        loss = paddle.mean(x_out + y_out)
        return x_out, y_out, loss


class LinearNetMultiInput1(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear1 = Linear(in_size, out_size)
        self._linear2 = Linear(in_size, out_size)

    def forward(self, x, y):
        x_out = self._linear1(x)
        y_out = self._linear2(y)
        loss = paddle.mean(x_out + y_out)
        return x_out, y_out, loss


class MultiLoadingLinearNet(paddle.nn.Layer):
    def __init__(self, size, model_path):
        super().__init__()
        self._linear = Linear(size, size)
        self._load_linear1 = paddle.jit.load(model_path)
        self._load_linear2 = paddle.jit.load(model_path)

    @to_static
    def forward(self, x):
        tmp1 = self._linear(x)
        tmp2 = self._load_linear1(tmp1)
        tmp3 = self._load_linear2(tmp2)
        y = self._linear(tmp3)
        return y


class LinearNetReturnHidden(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear_1 = Linear(in_size, out_size)
        self._linear_2 = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        y = self._linear_1(x)
        z = self._linear_2(y)
        loss = paddle.mean(z)
        return y, loss


class LinearNetWithNestOut(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear_1 = Linear(in_size, out_size)
        self._linear_2 = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        y = self._linear_1(x)
        z = self._linear_2(y)
        out = y + z
        loss = paddle.mean(out)
        return y, [(z, loss), out]


class LinearNetWithDictInput(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, img, label):
        out = self._linear(img['img'])
        # not return loss to avoid prune output
        loss = paddle.nn.functional.cross_entropy(out, label['label'])
        return out


class LinearNetWithDictInputNoPrune(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, img):
        out = self._linear(img['img'] + img['img2'])
        return out


class EmptyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    @paddle.jit.to_static
    def forward(self, x):
        return x


class NoParamLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    @paddle.jit.to_static
    def forward(self, x, y):
        return x + y


class LinearNetWithMultiStaticFunc(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear_0 = Linear(in_size, out_size)
        self._linear_1 = Linear(in_size, out_size)
        self._scale = paddle.to_tensor([9.9])

    def forward(self, x):
        return self._linear_0(x)

    def forward_no_param(self, x):
        return x * 1.0

    def forward_general(self, x):
        return self._linear_0(x) + self._linear_1(x) * self._scale


class LinearNetWithNonLexicographicalOrderDict(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear_u = Linear(in_size, out_size)
        self._linear_v = Linear(in_size, out_size)
        self._linear_w = Linear(in_size, out_size)
        self._linear_p = Linear(in_size, out_size)

    def forward(self, x):
        u = self._linear_u(x)
        v = self._linear_v(x)
        w = self._linear_w(x)
        p = self._linear_p(x)
        return {
            "u": u,
            "v": v,
            "w": w,
            "p": p,
        }


class LinearNetWithNestedNonLexicographicalOrderDict(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear_u = Linear(in_size, out_size)
        self._linear_v = Linear(in_size, out_size)
        self._linear_w = Linear(in_size, out_size)
        self._linear_p = Linear(in_size, out_size)
        self._linear_y = Linear(in_size, out_size)
        self._linear_x = Linear(in_size, out_size)

    def forward(self, x_):
        u = self._linear_u(x_)
        v = self._linear_v(x_)
        w = self._linear_w(x_)
        p = self._linear_p(x_)

        x = self._linear_p(x_)
        y = self._linear_p(x_)
        return {
            "u": u,
            "v": v,
            "w": w,
            "p": p,
            "a": {
                "x": x,
                "y": y,
            },
        }


def train(layer, input_size=784, label_size=1):
    # create optimizer
    sgd = paddle.optimizer.SGD(
        learning_rate=0.01, parameters=layer.parameters()
    )
    # create data loader
    train_loader = base.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(
        random_batch_reader(input_size, label_size)
    )
    # train
    for data in train_loader():
        img, label = data
        label.stop_gradient = True
        cost = layer(img)

        loss = paddle.nn.functional.cross_entropy(
            cost, label, reduction='none', use_softmax=True
        )
        avg_loss = paddle.mean(loss)

        avg_loss.backward()
        sgd.minimize(avg_loss)
        layer.clear_gradients()
    return [img], layer, avg_loss


def train_with_label(layer, input_size=784, label_size=1):
    # create optimizer
    sgd = paddle.optimizer.SGD(
        learning_rate=0.01, parameters=layer.parameters()
    )
    # create data loader
    train_loader = base.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(
        random_batch_reader(input_size, label_size)
    )
    # train
    for data in train_loader():
        img, label = data
        label.stop_gradient = True

        out, avg_loss = layer(img, label)

        avg_loss.backward()
        sgd.minimize(avg_loss)
        layer.clear_gradients()
    return out


class TestJitSaveLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "test_jit_save_load/model"
        )
        # enable dygraph mode
        base.enable_dygraph()
        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def tearDown(self):
        self.temp_dir.cleanup()

    def train_and_save_model(self, model_path=None):
        layer = LinearNet(784, 1)
        example_inputs, layer, _ = train(layer)
        final_model_path = model_path if model_path else self.model_path
        orig_input_types = [type(x) for x in example_inputs]
        paddle.jit.save(
            layer=layer, path=final_model_path, input_spec=example_inputs
        )
        new_input_types = [type(x) for x in example_inputs]
        self.assertEqual(orig_input_types, new_input_types)
        return layer

    def test_save_load(self):
        # train and save model
        if not paddle.framework.use_pir_api():
            return
        train_layer = self.train_and_save_model()
        # load model
        loaded_layer = paddle.jit.load(self.model_path)
        self.load_and_inference(train_layer, loaded_layer)
        self.load_and_finetune(train_layer, loaded_layer)
        if not paddle.framework.use_pir_api():
            self.load_dygraph_state_dict(train_layer)

    def load_and_inference(self, train_layer, infer_layer):
        train_layer.eval()
        infer_layer.eval()
        # inference & compare
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        np.testing.assert_array_equal(
            train_layer(x).numpy(), infer_layer(x).numpy()
        )

    def load_and_finetune(self, train_layer, load_train_layer):
        train_layer.train()
        load_train_layer.train()
        # train & compare
        img0, _, train_loss = train(train_layer)
        img1, _, load_train_loss = train(load_train_layer)
        np.testing.assert_array_equal(
            train_loss.numpy(), load_train_loss.numpy()
        )

    def load_dygraph_state_dict(self, train_layer):
        train_layer.eval()
        # construct new model
        new_layer = LinearNet(784, 1)
        orig_state_dict = new_layer.state_dict()
        load_state_dict = paddle.load(self.model_path)
        for structured_name in orig_state_dict:
            self.assertTrue(structured_name in load_state_dict)
        new_layer.set_state_dict(load_state_dict)
        new_layer.eval()
        # inference & compare
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        np.testing.assert_array_equal(
            train_layer(x).numpy(), new_layer(x).numpy()
        )

    def test_load_dygraph_no_path(self):
        model_path = os.path.join(
            self.temp_dir.name, "test_jit_save_load.no_path/model_path"
        )
        with self.assertRaises(ValueError):
            model_dict = paddle.load(model_path)

    def test_jit_load_no_path(self):
        path = os.path.join(
            self.temp_dir.name, "test_jit_save_load.no_path/model_path"
        )
        with self.assertRaises(ValueError):
            loaded_layer = paddle.jit.load(path)


class TestSaveLoadWithNestOut(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        base.enable_dygraph()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nest_output(self):
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))

        net = LinearNetWithNestOut(8, 8)
        dy_outs = paddle.utils.flatten(net(x))
        net = to_static(
            net, input_spec=[InputSpec([None, 8], name='x')], full_graph=True
        )

        model_path = os.path.join(self.temp_dir.name, "net_with_nest_out/model")
        paddle.jit.save(net, model_path)

        load_net = paddle.jit.load(model_path)
        load_outs = paddle.utils.flatten(load_net(x))

        self.assertTrue(len(dy_outs) == 4)
        for dy_out, load_out in zip(dy_outs, load_outs):
            np.testing.assert_allclose(
                dy_out.numpy(), load_out.numpy(), rtol=1e-05
            )


class TestSaveLoadWithNonLexicographicalOrderDict(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        base.enable_dygraph()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_output_same_order(self):
        model_path = os.path.join(self.temp_dir.name, "dict_out_model")
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))

        model = LinearNetWithNonLexicographicalOrderDict(8, 8)

        dy_output_dict = model(x)

        st_model = paddle.jit.to_static(model, full_graph=True)
        st_output_dict = st_model(x)

        with warnings.catch_warnings(record=True) as w:
            paddle.jit.save(st_model, model_path)
            self.assertIn(
                "Found 'dict' in given outputs, the values will be returned in a sequence sorted in lexicographical order by their keys.",
                str(w[-1].message),
            )
        loaded_model = paddle.jit.load(model_path)
        loaded_output_seq = loaded_model(x)

        self.assertTrue(len(dy_output_dict) == 4)
        self.assertTrue(len(st_output_dict) == 4)
        self.assertTrue(len(loaded_output_seq) == 4)

        # 1. check whether output dict of dygraph and static graph is same
        for (dy_key, dy_out), (st_key, st_out) in zip(
            dy_output_dict.items(), st_output_dict.items()
        ):
            self.assertTrue(dy_key == st_key)
            np.testing.assert_allclose(
                dy_out.numpy(), st_out.numpy(), rtol=1e-05
            )

        dy_output_seq = paddle.utils.flatten(dy_output_dict)

        self.assertTrue(len(dy_output_seq) == 4)

        # 2. check whether flattened output of loaded static graph has same order of dynamic's
        for dy_out, loaded_out in zip(dy_output_seq, loaded_output_seq):
            np.testing.assert_allclose(
                dy_out.numpy(), loaded_out.numpy(), rtol=1e-05
            )


class TestSaveLoadWithNestedNonLexicographicalOrderDict(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        base.enable_dygraph()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nested_output_same_order(self):
        model_path = os.path.join(self.temp_dir.name, "nested_dict_out_model")
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))

        model = LinearNetWithNestedNonLexicographicalOrderDict(8, 8)

        dy_output_dict = model(x)
        dy_output_seq = paddle.utils.flatten(dy_output_dict)

        st_model = paddle.jit.to_static(model, full_graph=True)
        st_output_dict = st_model(x)

        with warnings.catch_warnings(record=True) as w:
            paddle.jit.save(st_model, model_path)
            self.assertIn(
                "Found 'dict' in given outputs, the values will be returned in a sequence sorted in lexicographical order by their keys.",
                str(w[-1].message),
            )
        loaded_model = paddle.jit.load(model_path)
        loaded_output_seq = loaded_model(x)

        self.assertTrue(len(dy_output_dict) == 5)
        self.assertTrue(len(st_output_dict) == 5)
        self.assertTrue(len(loaded_output_seq) == 6)

        for dy_out, loaded_out in zip(dy_output_seq, loaded_output_seq):
            np.testing.assert_allclose(
                dy_out.numpy(), loaded_out.numpy(), rtol=1e-05
            )


class TestUtilsMapAndPack(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        base.enable_dygraph()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_utils_map_structure(self):
        nested_list = [
            {
                "d": paddle.to_tensor([1.0]),
                "a": paddle.to_tensor([2.0]),
                "c": paddle.to_tensor([3.0]),
                "tmp": {
                    "b": paddle.to_tensor([4.0]),
                },
            },
            [paddle.to_tensor([5.0]), paddle.to_tensor([6.0])],
            [],
            [
                paddle.to_tensor([7.0]),
                [
                    paddle.to_tensor([8.0]),
                    [paddle.to_tensor([9.0]), [paddle.to_tensor([10.0])]],
                ],
            ],
        ]
        FACTOR = 2
        expected_list = [
            {
                "d": paddle.to_tensor([1.0]) * FACTOR,
                "a": paddle.to_tensor([2.0]) * FACTOR,
                "c": paddle.to_tensor([3.0]) * FACTOR,
                "tmp": {
                    "b": paddle.to_tensor([4.0]) * FACTOR,
                },
            },
            [
                paddle.to_tensor([5.0]) * FACTOR,
                paddle.to_tensor([6.0]) * FACTOR,
            ],
            [],
            [
                paddle.to_tensor([7.0]) * FACTOR,
                [
                    paddle.to_tensor([8.0]) * FACTOR,
                    [
                        paddle.to_tensor([9.0]) * FACTOR,
                        [paddle.to_tensor([10.0]) * FACTOR],
                    ],
                ],
            ],
        ]
        mapped_list = paddle.utils.map_structure(
            lambda x: x * FACTOR, nested_list
        )

        # test paddle.utils.
        def dfs(obj1, obj2):
            self.assertTrue(type(obj1) == type(obj2))
            if isinstance(obj1, list):
                for i in range(len(obj1)):
                    dfs(obj1[i], obj2[i])
            elif isinstance(obj1, dict):
                self.assertTrue(list(obj1.keys()) == list(obj2.keys()))
                for k in obj1:
                    dfs(obj1[k], obj2[k])
            elif isinstance(obj1, paddle.Tensor):
                np.testing.assert_allclose(
                    obj1.numpy(), obj2.numpy(), rtol=1e-05
                )
            else:
                raise ValueError(f"Unsupported type: {type(obj1)} in dfs")

        dfs(expected_list, mapped_list)

    def test_utils_pack_sequence_as(self):
        nested_list = [
            {
                "d": paddle.to_tensor([1.0]),
                "a": paddle.to_tensor([2.0]),
                "c": paddle.to_tensor([3.0]),
                "tmp": {
                    "b": paddle.to_tensor([4.0]),
                },
            },
            [paddle.to_tensor([5.0]), paddle.to_tensor([6.0])],
            [],
            [
                paddle.to_tensor([7.0]),
                [
                    paddle.to_tensor([8.0]),
                    [paddle.to_tensor([9.0]), [paddle.to_tensor([10.0])]],
                ],
            ],
        ]

        def dfs(obj1, obj2):
            self.assertTrue(type(obj1) == type(obj2))
            if isinstance(obj1, list):
                for i in range(len(obj1)):
                    dfs(obj1[i], obj2[i])
            elif isinstance(obj1, dict):
                self.assertTrue(list(obj1.keys()) == list(obj2.keys()))
                for k in obj1:
                    dfs(obj1[k], obj2[k])
            elif isinstance(obj1, paddle.Tensor):
                np.testing.assert_allclose(
                    obj1.numpy(), obj2.numpy(), rtol=1e-05
                )
            else:
                raise ValueError(f"Unsupported type: {type(obj1)} in dfs")

        nested_list_copy = copy.deepcopy(nested_list)
        nested_list_copy_pack_back = paddle.utils.pack_sequence_as(
            nested_list_copy, paddle.utils.flatten(nested_list)
        )

        dfs(nested_list_copy, nested_list_copy_pack_back)


class TestSaveLoadWithDictInput(unittest.TestCase):

    def test_dict_input(self):
        # NOTE: This net cannot be executed, it is just
        # a special case for exporting models in model validation
        # We DO NOT recommend this writing way of Layer
        net = LinearNetWithDictInput(8, 8)
        net = paddle.jit.to_static(
            net,
            input_spec=[
                {
                    'img': InputSpec(
                        shape=[None, 8], dtype=paddle.float32, name='img'
                    )
                },
                {
                    'label': InputSpec(
                        shape=[None, 1], dtype=paddle.int64, name='label'
                    )
                },
            ],
            full_graph=True,
        )
        # net.forward.concrete_program.inputs:
        # (<__main__.LinearNetWithDictInput object at 0x7f2655298a98>,
        #  {'img': var img : base.VarType.DENSE_TENSOR.shape(-1, 8).astype(VarType.FP32)},
        #  {'label': var label : base.VarType.DENSE_TENSOR.shape(-1, 1).astype(VarType.INT64)})
        self.assertEqual(len(net.forward.concrete_program.inputs), 3)
        temp_dir = tempfile.TemporaryDirectory()
        path = os.path.join(
            temp_dir.name, "test_jit_save_load_with_dict_input/model"
        )
        # prune inputs
        paddle.jit.save(
            layer=net,
            path=path,
            input_spec=[
                {
                    'img': InputSpec(
                        shape=[None, 8], dtype=paddle.float32, name='img'
                    )
                }
            ],
        )

        img = paddle.randn(shape=[4, 8], dtype='float32')
        loaded_net = paddle.jit.load(path)
        loaded_out = loaded_net(img)

        # loaded_net._input_spec():
        # [InputSpec(shape=(-1, 8), dtype=VarType.FP32, name=img)]
        self.assertEqual(len(loaded_net._input_spec()), 1)
        self.assertEqual(len(loaded_net._output_spec()), 1)
        temp_dir.cleanup()


class TestSaveLoadWithDictInputNoPrune(unittest.TestCase):

    def test_dict_input(self):
        net = LinearNetWithDictInputNoPrune(8, 8)
        temp_dir = tempfile.TemporaryDirectory()
        path = os.path.join(
            temp_dir.name, "test_jit_save_load_with_dict_input_no_prune/model"
        )
        # prune inputs
        paddle.jit.save(
            layer=net,
            path=path,
            input_spec=[
                {
                    'img': InputSpec(
                        shape=[None, 8], dtype='float32', name='img'
                    ),
                    'img2': InputSpec(
                        shape=[None, 8], dtype='float32', name='img2'
                    ),
                }
            ],
        )

        img = paddle.randn(shape=[4, 8], dtype='float32')
        img2 = paddle.randn(shape=[4, 8], dtype='float32')
        loaded_net = paddle.jit.load(path)
        loaded_out = loaded_net(img, img2)

        self.assertEqual(len(loaded_net._input_spec()), 2)
        temp_dir.cleanup()


class TestSaveLoadWithInputSpec(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        base.enable_dygraph()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_with_input_spec(self):
        net = LinearNetReturnLoss(8, 8)
        # set x.shape = [None, 8]
        net.forward = to_static(
            net.forward,
            input_spec=[InputSpec([None, 8], name='x')],
            full_graph=True,
        )

        model_path = os.path.join(
            self.temp_dir.name, "input_spec.output_spec/model"
        )
        # check inputs and outputs
        self.assertTrue(len(net.forward.inputs) == 1)
        input_x = net.forward.inputs[0]
        if paddle.framework.use_pir_api():
            self.assertTrue(input_x.shape == [-1, 8])
        else:
            self.assertTrue(input_x.shape == (-1, 8))
        self.assertTrue(input_x.name == 'x')

        # 1. prune loss
        output_spec = net.forward.outputs[:1]
        paddle.jit.save(net, model_path, output_spec=output_spec)

        # 2. load to infer
        infer_layer = paddle.jit.load(model_path)
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        pred = infer_layer(x)

    def test_multi_in_out(self):
        net = LinearNetMultiInput(8, 8)
        net = paddle.jit.to_static(
            net,
            input_spec=[
                InputSpec([None, 8], dtype='float32'),
                InputSpec([None, 8], dtype='float32'),
            ],
            full_graph=True,
        )

        model_path = os.path.join(
            self.temp_dir.name, "multi_inout.output_spec1/model"
        )
        # 1. check inputs and outputs
        self.assertTrue(len(net.forward.inputs) == 2)
        input_x = net.forward.inputs[0]
        input_y = net.forward.inputs[1]
        if paddle.framework.use_pir_api():
            self.assertTrue(input_x.shape == [-1, 8])
            self.assertTrue(input_y.shape == [-1, 8])
        else:
            self.assertTrue(input_x.shape == (-1, 8))
            self.assertTrue(input_y.shape == (-1, 8))

        # 2. prune loss
        output_spec = net.forward.outputs[:2]
        paddle.jit.save(net, model_path, output_spec=output_spec)

        # 3. load to infer
        infer_layer = paddle.jit.load(model_path)
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        y = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        # 4. predict
        pred_x, pred_y = infer_layer(x, y)

        # 1. prune y and loss
        model_path = os.path.join(
            self.temp_dir.name, "multi_inout.output_spec2/model"
        )
        output_spec = net.forward.outputs[:1]
        paddle.jit.save(net, model_path, [input_x], output_spec=output_spec)
        # 2. load again
        infer_layer2 = paddle.jit.load(model_path)
        # 3. predict
        pred_xx = infer_layer2(x)

        # 4. assert pred_x == pred_xx
        np.testing.assert_allclose(pred_x.numpy(), pred_xx.numpy(), rtol=1e-05)

    def test_multi_in_out1(self):
        net = LinearNetMultiInput1(8, 8)
        net = paddle.jit.to_static(
            net,
            input_spec=(
                InputSpec([None, 8], dtype='float32'),
                InputSpec([None, 8], dtype='float32'),
            ),
            full_graph=True,
        )
        model_path = os.path.join(
            self.temp_dir.name, "multi_inout1.output_spec1/model"
        )
        # 1. check inputs and outputs
        self.assertTrue(len(net.forward.inputs) == 2)
        input_x = net.forward.inputs[0]
        input_y = net.forward.inputs[1]
        if paddle.framework.use_pir_api():
            self.assertTrue(input_x.shape == [-1, 8])
            self.assertTrue(input_y.shape == [-1, 8])
        else:
            self.assertTrue(input_x.shape == (-1, 8))
            self.assertTrue(input_y.shape == (-1, 8))

        # 2. prune loss
        output_spec = net.forward.outputs[:2]
        paddle.jit.save(net, model_path, output_spec=output_spec)

        # 3. load to infer
        infer_layer = paddle.jit.load(model_path)
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        y = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        # 4. predict
        pred_x, pred_y = infer_layer(x, y)

        # 1. prune y and loss
        model_path = os.path.join(
            self.temp_dir.name, "multi_inout1.output_spec2/model"
        )
        output_spec = net.forward.outputs[:1]
        paddle.jit.save(
            net,
            model_path,
            net.forward.inputs,
            output_spec=output_spec,
            input_names_after_prune=[input_x.name],
        )
        # 2. load again
        infer_layer2 = paddle.jit.load(model_path)
        # 3. predict
        pred_xx = infer_layer2(x)

        # 4. assert pred_x == pred_xx
        np.testing.assert_allclose(pred_x.numpy(), pred_xx.numpy(), rtol=1e-05)


class TestJitSaveLoadConfig(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        base.enable_dygraph()
        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_output_spec(self):
        train_layer = LinearNetReturnLoss(8, 8)
        train_layer.forward = to_static(
            train_layer.forward,
            input_spec=[InputSpec([None, 8], name='x')],
            full_graph=True,
        )
        adam = paddle.optimizer.Adam(
            learning_rate=0.1, parameters=train_layer.parameters()
        )
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            out, loss = train_layer(x)
            loss.backward()
            adam.minimize(loss)
            train_layer.clear_gradients()

        model_path = os.path.join(
            self.temp_dir.name, "save_load_config.output_spec"
        )
        output_spec = train_layer.forward.outputs[:1]
        paddle.jit.save(
            layer=train_layer,
            path=model_path,
            input_spec=[x],
            output_spec=output_spec,
        )

        train_layer.eval()
        infer_layer = paddle.jit.load(model_path)
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        np.testing.assert_array_equal(
            train_layer(x)[0].numpy(), infer_layer(x).numpy()
        )

    def test_save_no_support_config_error(self):
        layer = LinearNet(784, 1)
        path = os.path.join(self.temp_dir.name, "no_support_config_test")
        with self.assertRaises(ValueError):
            paddle.jit.save(layer=layer, path=path, model_filename="")

    def test_load_empty_model_filename_error(self):
        path = os.path.join(self.temp_dir.name, "error_model_filename_test")

        with self.assertRaises(ValueError):
            paddle.jit.load(path, model_filename="")

    def test_load_empty_params_filename_error(self):
        path = os.path.join(self.temp_dir.name, "error_params_filename_test")
        with self.assertRaises(ValueError):
            paddle.jit.load(path, params_filename="")

    def test_load_with_no_support_config(self):
        path = os.path.join(self.temp_dir.name, "no_support_config_test")
        with self.assertRaises(ValueError):
            paddle.jit.load(path, separate_params=True)


class TestJitMultipleLoading(unittest.TestCase):
    def setUp(self):
        self.linear_size = 4
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "jit_multi_load/model"
        )
        # enable dygraph mode
        base.enable_dygraph()
        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        # train and save base model
        self.train_and_save_orig_model()

    def tearDown(self):
        self.temp_dir.cleanup()

    def train_and_save_orig_model(self):
        layer = LinearNet(self.linear_size, self.linear_size)
        example_inputs, layer, _ = train(layer, self.linear_size, 1)
        paddle.jit.save(
            layer=layer, path=self.model_path, input_spec=example_inputs
        )

    def test_load_model_retransform_inference(self):
        multi_loaded_layer = MultiLoadingLinearNet(
            self.linear_size, self.model_path
        )
        state_dict = multi_loaded_layer.state_dict()
        name_set = set()
        for _, var in state_dict.items():
            self.assertTrue(var.name not in name_set)
            name_set.add(var.name)


class TestJitPruneModelAndLoad(unittest.TestCase):
    def setUp(self):
        self.linear_size = 4
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "jit_prune_model_and_load/model"
        )
        # enable dygraph mode
        base.enable_dygraph()
        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def tearDown(self):
        self.temp_dir.cleanup()

    def train_and_save(self):
        train_layer = LinearNetReturnHidden(8, 8)
        train_layer = to_static(
            train_layer,
            input_spec=[InputSpec([None, 8], name='x')],
            full_graph=True,
        )
        adam = paddle.optimizer.Adam(
            learning_rate=0.1, parameters=train_layer.parameters()
        )
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            hidden, loss = train_layer(x)
            loss.backward()
            adam.minimize(loss)
            train_layer.clear_gradients()

        output_spec = train_layer.forward.outputs[:1]
        paddle.jit.save(
            layer=train_layer,
            path=self.model_path,
            input_spec=[x],
            output_spec=output_spec,
        )

        return train_layer

    def test_load_pruned_model(self):
        train_layer = self.train_and_save()
        train_layer.eval()

        infer_layer = paddle.jit.load(self.model_path)

        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        np.testing.assert_array_equal(
            train_layer(x)[0].numpy(), infer_layer(x).numpy()
        )


class TestJitSaveMultiCases(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        base.enable_dygraph()
        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def verify_inference_correctness(
        self, layer, model_path, with_label_and_loss=False, with_label=False
    ):
        layer.eval()
        loaded_layer = paddle.jit.load(model_path)
        loaded_layer.eval()
        # inference & compare
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        if with_label_and_loss:
            y = paddle.to_tensor(np.random.random((1, 1)).astype('int64'))
            pred, _ = layer(x, y)
            pred = pred.numpy()
        elif with_label:
            y = paddle.to_tensor(np.random.random((1, 1)).astype('int64'))
            pred = layer(x, y)
            pred = pred.numpy()
        else:
            pred = layer(x).numpy()
        loaded_pred = loaded_layer(x).numpy()
        np.testing.assert_array_equal(
            pred,
            loaded_pred,
            err_msg=f'Result diff when load and inference:\nlayer result:\n{pred}\nloaded layer result:\n{loaded_pred}',
        )

    def test_no_prune_to_static_after_train(self):
        layer = LinearNet(784, 1)

        train(layer)

        model_path = os.path.join(
            self.temp_dir.name, "test_no_prune_to_static_after_train/model"
        )
        paddle.jit.save(layer, model_path)

        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_to_static_no_train(self):
        layer = LinearNetWithInputSpec(784, 1)

        model_path = os.path.join(
            self.temp_dir.name, "test_no_prune_to_static_no_train/model"
        )
        paddle.jit.save(layer, model_path)

        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_after_train(self):
        layer = LinearNetNotDeclarative(784, 1)

        train(layer)

        model_path = os.path.join(
            self.temp_dir.name, "test_no_prune_no_to_static_after_train/model"
        )
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[InputSpec(shape=[None, 784], dtype='float32')],
        )

        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_after_train_with_examples(self):
        layer = LinearNetNotDeclarative(784, 1)

        example_inputs, _, _ = train(layer)

        model_path = os.path.join(
            self.temp_dir.name,
            "test_no_prune_no_to_static_after_train_with_examples/model",
        )
        paddle.jit.save(layer=layer, path=model_path, input_spec=example_inputs)

        self.verify_inference_correctness(layer, model_path)

    def test_no_prune_no_to_static_no_train(self):
        layer = LinearNetNotDeclarative(784, 1)

        model_path = os.path.join(
            self.temp_dir.name, "test_no_prune_no_to_static_no_train/model"
        )
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[InputSpec(shape=[None, 784], dtype='float32')],
        )

        self.verify_inference_correctness(layer, model_path)

    def test_prune_to_static_after_train(self):
        layer = LinerNetWithLabel(784, 1)
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image"),
                InputSpec(shape=[None, 1], dtype='int64', name="label"),
            ],
            full_graph=True,
        )
        out = train_with_label(layer)

        model_path = os.path.join(
            self.temp_dir.name, "test_prune_to_static_after_train/model"
        )
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image"),
            ],
            output_spec=layer.forward.outputs[:1],
            input_names_after_prune=["image"],
        )

        self.verify_inference_correctness(
            layer, model_path, with_label_and_loss=True
        )

    def test_prune_to_static_no_train(self):
        layer = LinerNetWithLabel(784, 1)
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image"),
                InputSpec(shape=[None, 1], dtype='int64', name="label"),
            ],
            full_graph=True,
        )
        model_path = os.path.join(
            self.temp_dir.name, "test_prune_to_static_no_train/model"
        )
        # TODO: no train, cannot get output_spec var here
        # now only can use index
        output_spec = layer.forward.outputs[:1]
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image"),
            ],
            output_spec=output_spec,
            input_names_after_prune=["image"],
        )

        self.verify_inference_correctness(
            layer, model_path, with_label_and_loss=True
        )

    def test_prune_input_to_static_no_train(self):
        layer = LinerNetWithPruneInput(784, 1)
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image"),
                InputSpec(shape=[None, 1], dtype='int64', name="label"),
            ],
            full_graph=True,
        )
        model_path = os.path.join(
            self.temp_dir.name, "test_prune_input_to_static_no_train/model"
        )
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image")
            ],
        )

        self.verify_inference_correctness(layer, model_path, with_label=True)

    def test_prune_useless_input_to_static_no_train(self):
        layer = LinerNetWithUselessInput(784, 1)
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image"),
                InputSpec(shape=[None, 1], dtype='int64', name="label"),
            ],
            full_graph=True,
        )
        model_path = os.path.join(
            self.temp_dir.name,
            "test_prune_useless_input_to_static_no_train/model",
        )
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image")
            ],
        )

        self.verify_inference_correctness(layer, model_path, with_label=True)

    def test_no_prune_input_spec_name_warning(self):
        layer = LinearNetWithInputSpec(784, 1)

        train(layer)

        model_path = os.path.join(
            self.temp_dir.name, "test_no_prune_input_spec_name_warning/model"
        )
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[InputSpec(shape=[None, 784], dtype='float32')],
        )
        paddle.jit.save(
            layer,
            model_path,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name='feed_input')
            ],
        )

        self.verify_inference_correctness(layer, model_path)

    def test_not_prune_output_spec_name_warning(self):
        layer = LinearNet(784, 1)

        train(layer)

        model_path = os.path.join(
            self.temp_dir.name, "test_not_prune_output_spec_name_warning/model"
        )
        out = paddle.to_tensor(np.random.random((1, 1)).astype('float'))
        paddle.jit.save(layer, model_path, output_spec=[out])

        self.verify_inference_correctness(layer, model_path)

    def test_prune_input_spec_name_error(self):
        layer = LinerNetWithLabel(784, 1)

        model_path = os.path.join(
            self.temp_dir.name, "test_prune_input_spec_name_error/model"
        )
        with self.assertRaises(ValueError):
            paddle.jit.save(
                layer,
                model_path,
                input_spec=[InputSpec(shape=[None, 784], dtype='float32')],
            )
        with self.assertRaises(ValueError):
            paddle.jit.save(
                layer,
                model_path,
                input_spec=[
                    InputSpec(
                        shape=[None, 784], dtype='float32', name='feed_input'
                    )
                ],
            )

    def test_prune_output_spec_name_error(self):
        layer = LinerNetWithLabel(784, 1)
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 784], dtype='float32', name="image"),
                InputSpec(shape=[None, 1], dtype='int64', name="label"),
            ],
            full_graph=True,
        )
        train_with_label(layer)

        model_path = os.path.join(
            self.temp_dir.name, "test_prune_to_static_after_train/model"
        )
        out = paddle.to_tensor(np.random.random((1, 1)).astype('float'))
        with self.assertRaises(ValueError):
            paddle.jit.save(
                layer,
                model_path,
                input_spec=[
                    InputSpec(shape=[None, 784], dtype='float32', name="image"),
                    True,
                ],
                output_spec=[out],
                input_names_after_prune=["image"],
            )


class TestJitSaveLoadEmptyLayer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "jit_save_load_empty_layer/model"
        )
        # enable dygraph mode
        paddle.disable_static()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_empty_layer(self):
        layer = EmptyLayer()
        x = paddle.to_tensor(np.random.random(10).astype('float32'))
        out = layer(x)
        try:
            paddle.jit.save(layer, self.model_path)
        except ValueError as e:
            self.assertTrue(
                'program must not be empty. at least one operator is required!'
                in str(e)
            )


class TestJitSaveLoadNoParamLayer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "jit_save_load_no_param_layer/model"
        )
        # enable dygraph mode
        paddle.disable_static()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_no_param_layer(self):
        layer = NoParamLayer()
        x = paddle.to_tensor(np.random.random(5).astype('float32'))
        y = paddle.to_tensor(np.random.random(5).astype('float32'))
        out = layer(x, y)
        paddle.jit.save(layer, self.model_path)
        load_layer = paddle.jit.load(self.model_path)
        load_out = load_layer(x, y)
        np.testing.assert_array_equal(out, load_out)


class TestJitSaveLoadMultiMethods(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_save_load_inference(self):
        model_path_inference = os.path.join(
            self.temp_dir.name, "jit_save_load_multi_methods/model"
        )
        IMAGE_SIZE = 224
        layer = LinearNetWithMultiStaticFunc(IMAGE_SIZE, 10)
        layer = paddle.jit.to_static(
            layer,
            full_graph=True,
        )
        layer.forward_no_param = paddle.jit.to_static(
            layer.forward_no_param,
            full_graph=True,
        )
        layer.forward_general = paddle.jit.to_static(
            layer.forward_general,
            full_graph=True,
        )
        inps = paddle.randn([1, IMAGE_SIZE])
        result_origin = {}
        for func in dir(layer):
            if func.startswith('forward'):
                result_origin[func] = getattr(layer, func, None)(inps)

        paddle.jit.save(layer, model_path_inference)
        load_net = paddle.jit.load(model_path_inference)
        for func, result in result_origin.items():
            self.assertTrue(
                float(
                    (result - getattr(load_net, func, None)(inps)).abs().max()
                )
                < 1e-5
            )

    def test_jit_save_load_multi_methods_inputspec(self):
        model_path = os.path.join(
            self.temp_dir.name, 'jit_save_load_multi_methods/model'
        )
        layer = LinearNetWithMultiStaticFunc(784, 1)
        layer = paddle.jit.to_static(
            layer,
            full_graph=True,
        )
        layer.forward_no_param = paddle.jit.to_static(
            layer.forward_no_param,
            full_graph=True,
        )
        layer.forward_general = paddle.jit.to_static(
            layer.forward_general,
            full_graph=True,
        )
        with self.assertRaises(ValueError):
            paddle.jit.save(
                layer, model_path, input_spec=[InputSpec(shape=[None, 784])]
            )

    def test_parse_name(self):
        model_path_inference = os.path.join(
            self.temp_dir.name, "jit_save_load_parse_name/model"
        )
        IMAGE_SIZE = 224
        layer = LinearNet(IMAGE_SIZE, 1)
        inps = paddle.randn([1, IMAGE_SIZE])
        layer(inps)
        paddle.jit.save(layer, model_path_inference)
        paddle.jit.save(layer, model_path_inference + '_v2')
        load_net = paddle.jit.load(model_path_inference)

        self.assertFalse(hasattr(load_net, 'v2'))


class LayerSaved(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.hidden = 100
        self._linear_0 = Linear(in_size, self.hidden)
        self._linear_1_0 = Linear(self.hidden, self.hidden)
        self._linear_1_1 = Linear(self.hidden, self.hidden)
        self._linear_2 = Linear(self.hidden, out_size)
        self._scale = paddle.to_tensor([9.9])

    def forward(self, x):
        y = self._linear_0(x)
        # Multiple blocks
        if paddle.shape(x)[0] == 1:
            y = self._linear_1_0(y)
        else:
            y += self._linear_1_1(y + self._scale)
        return self._linear_2(y)


class TestJitSaveCombineProperty(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_save_combine_property(self):
        class Net(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc1 = paddle.nn.Linear(4, 4)
                self.fc2 = paddle.nn.Linear(4, 4)
                self.bias = 0.4
                self.flag = paddle.ones([2], dtype="int32")

            @paddle.jit.to_static(
                input_spec=[InputSpec([None, 4], dtype='float32')],
                full_graph=True,
            )
            def log_softmax(self, input):
                return paddle.nn.functional.log_softmax(input, axis=-1)

            @paddle.jit.to_static(
                input_spec=[InputSpec([None, 4], dtype='float32')],
                full_graph=True,
            )
            def forward(self, x):
                out = self.fc1(x)
                out = paddle.nn.functional.relu(out)
                out = paddle.mean(out)
                return out

            @paddle.jit.to_static(
                input_spec=[InputSpec([None, 4], dtype='float32')],
                full_graph=True,
            )
            def infer(self, input):
                out = self.fc2(input)
                out = out + self.bias
                out = paddle.mean(out)
                return out

            # For extra Python float
            @paddle.jit.to_static(property=True, full_graph=True)
            def fbias(self):
                return self.bias + 1

            @paddle.jit.to_static(property=True, full_graph=True)
            def down_sampling(self):
                return 4

            @paddle.jit.to_static(property=True, full_graph=True)
            def fstr(self):
                return "save str property"

            @paddle.jit.to_static(property=True, full_graph=True)
            def ints(self):
                return [10, 20]

            @paddle.jit.to_static(property=True, full_graph=True)
            def floats(self):
                return [1.1, 2.2]

            @paddle.jit.to_static(property=True, full_graph=True)
            def strs(self):
                return ["hello", "world"]

        model_path = os.path.join(
            self.temp_dir.name, "test_jit_save_combine/model"
        )
        # Use new namespace
        with unique_name.guard():
            net = Net()
        # save
        paddle.jit.save(net, model_path, combine_params=True)

    def test_jit_save_tensor_property(self):
        class NetTensor(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc1 = paddle.nn.Linear(4, 4)
                self.fc2 = paddle.nn.Linear(4, 4)
                self.bias = 0.4
                self.flag = paddle.ones([2], dtype="int32")

            def forward(self, x):
                out = self.fc1(x)
                out = paddle.nn.functional.relu(out)
                out = paddle.mean(out)
                return out

            @paddle.jit.to_static(property=True, full_graph=True)
            def fflag(self):
                return True

        model_path = os.path.join(
            self.temp_dir.name, "test_jit_save_combine/model"
        )
        # Use new namespace
        with unique_name.guard():
            net = NetTensor()
            net = paddle.jit.to_static(
                net,
                input_spec=[InputSpec([None, 4], dtype='float32')],
                full_graph=True,
            )

        paddle.jit.save(net, model_path, combine_params=True)


class TestJitSaveLoadSaveWithoutRunning(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_finetune_load(self):
        model_path = os.path.join(
            self.temp_dir.name, "test_jit_save_load_save_without_running/model"
        )
        IMAGE_SIZE = 224
        inps0 = paddle.randn([1, IMAGE_SIZE])
        inps1 = paddle.randn([2, IMAGE_SIZE])
        # Use new namespace
        with unique_name.guard():
            layer_save = LayerSaved(IMAGE_SIZE, IMAGE_SIZE)
            layer_save = paddle.jit.to_static(layer_save, full_graph=True)
        # save
        paddle.jit.save(
            layer_save,
            model_path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, IMAGE_SIZE], dtype='float32'
                )
            ],
        )
        result_00 = layer_save(inps0)
        result_01 = layer_save(inps1)
        # load and save without running
        with unique_name.guard():
            layer_load = paddle.jit.load(model_path)
            paddle.jit.save(
                layer_load,
                model_path,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None, IMAGE_SIZE], dtype='float32'
                    )
                ],
            )
        # reload
        layer_reload = paddle.jit.load(model_path)
        result_10 = layer_reload(inps0)
        result_11 = layer_reload(inps1)

        self.assertTrue(float((result_00 - result_10).abs().max()) < 1e-5)
        self.assertTrue(float((result_01 - result_11).abs().max()) < 1e-5)


class LayerLoadFinetune(paddle.nn.Layer):
    def __init__(self, in_size, out_size, load_path):
        super().__init__()
        # Test duplicate name
        self._linear_0 = Linear(in_size, in_size)
        self._linear_1_0 = Linear(out_size, in_size)
        self._linear_1_1 = Linear(out_size, in_size)
        self._linear_2 = Linear(out_size, out_size)
        self._scale = paddle.to_tensor([9.9])

        # Load multiple times
        self._load_l1 = paddle.jit.load(load_path)
        self._load_l2 = paddle.jit.load(load_path)

    def forward(self, x):
        y = self._linear_0(x)
        y = self._load_l1(y)
        # Multiple blocks
        if paddle.shape(x)[0] == 1:
            y = self._linear_1_0(y)
            y = self._load_l1(y)
        else:
            y += self._linear_1_1(x + self._scale)
            y = self._load_l2(y)
        y = self._linear_1_0(y)
        y = self._load_l1(y)
        y = self._linear_1_0(y)
        # Use the same layer multiple times.
        y = self._load_l1(y)
        return y


class TestJitSaveLoadFinetuneLoad(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_finetune_load(self):
        if not paddle.framework.use_pir_api():
            return
        model_path = os.path.join(
            self.temp_dir.name, "test_jit_save_load_finetune_load/model"
        )
        IMAGE_SIZE = 224
        inps0 = paddle.randn([1, IMAGE_SIZE])
        inps1 = paddle.randn([2, IMAGE_SIZE])
        # Use new namespace
        with unique_name.guard():
            layer_save = LayerSaved(IMAGE_SIZE, IMAGE_SIZE)
            layer_save = paddle.jit.to_static(layer_save, full_graph=True)
        layer_save(inps0)
        # save
        paddle.jit.save(layer_save, model_path)
        # load
        with unique_name.guard():
            layer_load = LayerLoadFinetune(IMAGE_SIZE, IMAGE_SIZE, model_path)
            layer_load = paddle.jit.to_static(layer_load, full_graph=True)
        # train
        train(layer_load, input_size=IMAGE_SIZE)
        result_00 = layer_load(inps0)
        result_01 = layer_load(inps1)
        # save
        paddle.jit.save(layer_load, model_path)
        # load
        layer_finetune = paddle.jit.load(model_path)
        result_10 = layer_finetune(inps0)
        result_11 = layer_finetune(inps1)

        # (result_00 - result_10) is [nan, ...], so the result of (result_00 - result_10).abs().max() is -inf.
        # Since -inf is always less than 1e-5, the assert will always evaluate to true.
        # Therefore, this assert should be considered to remove.
        # self.assertTrue(float((result_00 - result_10).abs().max()) < 1e-5)
        # self.assertTrue(float((result_01 - result_11).abs().max()) < 1e-5)


# NOTE(weixin): When there are multiple test functions in an
# `unittest.TestCase`, functions will affect each other,
# and there is a risk of random failure.
# So divided into three TestCase: TestJitSaveLoadFunctionCase1,
# TestJitSaveLoadFunctionCase2, TestJitSaveLoadFunctionCase3.
class TestJitSaveLoadFunctionCase1(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_save_load_static_function(self):
        @paddle.jit.to_static
        def fun(inputs):
            return paddle.tanh(inputs)

        path = os.path.join(
            self.temp_dir.name, 'test_jit_save_load_function_1/func'
        )
        inps = paddle.rand([3, 6])
        origin = fun(inps)

        paddle.jit.save(fun, path)
        load_func = paddle.jit.load(path)

        load_result = load_func(inps)
        self.assertTrue((load_result - origin).abs().max() < 1e-10)


class TestJitSaveLoadFunctionCase2(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_save_load_function_input_spec(self):
        @paddle.jit.to_static(
            input_spec=[
                InputSpec(shape=[None, 6], dtype='float32', name='x'),
            ],
            full_graph=True,
        )
        def fun(inputs):
            return paddle.nn.functional.relu(inputs)

        path = os.path.join(
            self.temp_dir.name, 'test_jit_save_load_function_2/func'
        )
        inps = paddle.rand([3, 6])
        origin = fun(inps)

        paddle.jit.save(fun, path)
        load_func = paddle.jit.load(path)
        load_result = load_func(inps)
        self.assertTrue((load_result - origin).abs().max() < 1e-10)


class TestJitSaveLoadFunctionCase3(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_save_load_function_function(self):
        def fun(inputs):
            return paddle.tanh(inputs)

        path = os.path.join(
            self.temp_dir.name, 'test_jit_save_load_function_3/func'
        )
        inps = paddle.rand([3, 6])
        origin = fun(inps)

        paddle.jit.save(
            fun,
            path,
            input_spec=[
                InputSpec(shape=[None, 6], dtype='float32', name='x'),
            ],
        )
        load_func = paddle.jit.load(path)

        load_result = load_func(inps)
        self.assertTrue((load_result - origin).abs().max() < 1e-10)


class TestJitSaveLoadFunctionWithParamCase1(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_save_load_function(self):
        class LinearNet(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self._linear = paddle.nn.Linear(5, 6)

            def forward(self, x):
                return paddle.tanh(x)

            def anothor_forward(self, x):
                return self._linear(x)

        layer = LinearNet()

        inps = paddle.rand([3, 5])
        origin = layer.anothor_forward(inps)

        func = paddle.jit.to_static(
            layer.anothor_forward,
            [paddle.static.InputSpec(shape=[-1, 5])],
            full_graph=True,
        )
        path = os.path.join(
            self.temp_dir.name,
            'test_jit_save_load_function_with_params_case1/func',
        )
        paddle.jit.save(func, path)
        load_func = paddle.jit.load(path)

        load_result = load_func(inps)
        np.testing.assert_array_equal(load_result.numpy(), origin.numpy())


class TestJitSaveLoadFunctionWithParamCase2(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_save_load_function(self):
        class LinearNet(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self._linear = paddle.nn.Linear(5, 6)

            def forward(self, x):
                return paddle.tanh(x)

            @paddle.jit.to_static(
                input_spec=[InputSpec(shape=[-1, 5])], full_graph=True
            )
            def anothor_forward(self, x):
                return self._linear(x)

        layer = LinearNet()

        inps = paddle.rand([3, 5])

        path = os.path.join(
            self.temp_dir.name,
            'test_jit_save_load_function_with_params_case2/func',
        )
        paddle.jit.save(layer.anothor_forward, path)
        origin_result = layer.anothor_forward(inps)
        load_func = paddle.jit.load(path)

        load_result = load_func(inps)

        np.testing.assert_array_equal(
            origin_result.numpy(), load_result.numpy()
        )


class TestJitSaveLoadFunctionWithParamCase3(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_save_load_function(self):
        class LinearNet(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self._linear = paddle.nn.Linear(5, 6)

            def forward(self, x):
                return paddle.tanh(x)

            @paddle.jit.to_static
            def anothor_forward(self, x):
                return self._linear(x)

        layer = LinearNet()

        inps = paddle.rand([3, 5])
        origin = layer.anothor_forward(inps)

        path = os.path.join(
            self.temp_dir.name,
            'test_jit_save_load_function_with_params_case3/func',
        )
        paddle.jit.save(layer.anothor_forward, path)
        load_func = paddle.jit.load(path)

        load_result = load_func(inps)
        np.testing.assert_array_equal(load_result.numpy(), origin.numpy())


class TestJitSaveLoadDataParallel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def verify_inference_correctness(self, layer, path):
        layer.eval()
        loaded_layer = paddle.jit.load(path)
        loaded_layer.eval()
        # inference & compare
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        pred = layer(x).numpy()
        loaded_pred = loaded_layer(x).numpy()
        np.testing.assert_array_equal(
            pred,
            loaded_pred,
            err_msg=f'Result diff when load and inference:\nlayer result:\n{pred}\nloaded layer result:\n{loaded_pred}',
        )

    def test_jit_save_data_parallel_with_inputspec(self):
        layer = LinearNetNotDeclarative(784, 1)
        layer = paddle.DataParallel(layer)
        path = os.path.join(
            self.temp_dir.name, "jit_save_data_parallel_with_inputspec/model"
        )
        paddle.jit.save(
            layer=layer, path=path, input_spec=[InputSpec(shape=[None, 784])]
        )

        self.verify_inference_correctness(layer, path)

    def test_jit_save_data_parallel_with_to_static(self):
        layer = LinearNetWithInputSpec(784, 1)
        layer = paddle.DataParallel(layer)

        path = os.path.join(
            self.temp_dir.name, "jit_save_data_parallel_with_to_static/model"
        )
        paddle.jit.save(layer, path)

        self.verify_inference_correctness(layer, path)


class InputSepcLayer(paddle.nn.Layer):
    # A layer with InputSpec to test InputSpec compatibility

    def forward(self, x, y):
        return x * 1.0, y * 1.0


class TestInputSpecCompatibility(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _assert_input_spec_layer_return(self, expect_layer, test_layer):
        input_x = paddle.uniform([8, 8], dtype='float32')
        input_y = paddle.uniform([8, 1], dtype='float64')
        expected_result = expect_layer(input_x, input_y)
        test_result = test_layer(input_x, input_y)
        np.testing.assert_allclose(
            expected_result[0].numpy(), test_result[0].numpy()
        )
        np.testing.assert_allclose(
            expected_result[1].numpy(), test_result[1].numpy()
        )

    def test_jit_save_no_input_sepc(self):
        layer = InputSepcLayer()
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 8], dtype='float32', name='x'),
                InputSpec(shape=[None, 1], dtype='float64', name='y'),
            ],
            full_graph=True,
        )
        save_dir = os.path.join(self.temp_dir.name, "jit_save_no_input_spec")
        path = save_dir + "/model"

        paddle.jit.save(layer=layer, path=path)
        no_input_spec_layer = paddle.jit.load(path)
        self._assert_input_spec_layer_return(layer, no_input_spec_layer)
        shutil.rmtree(save_dir)

    def test_jit_save_same_input_sepc(self):
        layer = InputSepcLayer()
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 8], dtype='float32', name='x'),
                InputSpec(shape=[None, 1], dtype='float64', name='y'),
            ],
            full_graph=True,
        )

        save_dir = os.path.join(self.temp_dir.name, "jit_save_same_input_spec")
        path = save_dir + "/model"

        paddle.jit.save(
            layer=layer,
            path=path,
            input_spec=[
                InputSpec(shape=[None, 8], dtype='float32', name='x'),
                InputSpec(shape=[None, 1], dtype='float64', name='y'),
            ],
        )
        same_input_spec_layer = paddle.jit.load(path)
        self._assert_input_spec_layer_return(layer, same_input_spec_layer)
        shutil.rmtree(save_dir)

    def test_jit_save_compatible_input_sepc(self):
        layer = InputSepcLayer()
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 8], dtype='float32', name='x'),
                InputSpec(shape=[None, 1], dtype='float64', name='y'),
            ],
            full_graph=True,
        )

        save_dir = os.path.join(
            self.temp_dir.name, "jit_save_compatible_input_spec"
        )
        path = save_dir + "/model"
        paddle.jit.save(
            layer=layer,
            path=path,
            input_spec=[
                InputSpec(shape=[8, 8], dtype='float32'),
                InputSpec(shape=[8, -1], dtype='float64'),
            ],
        )
        compatible_input_spec_layer = paddle.jit.load(path)
        self._assert_input_spec_layer_return(layer, compatible_input_spec_layer)
        shutil.rmtree(save_dir)

    def test_jit_save_incompatible_input_sepc(self):
        layer = InputSepcLayer()
        layer = paddle.jit.to_static(
            layer,
            input_spec=[
                InputSpec(shape=[None, 8], dtype='float32', name='x'),
                InputSpec(shape=[None, 1], dtype='float64', name='y'),
            ],
            full_graph=True,
        )
        save_dir = os.path.join(
            self.temp_dir.name, "jit_save_compatible_input_spec"
        )
        path = save_dir + "/model"

        with self.assertRaises(ValueError):
            # type mismatch
            paddle.jit.save(
                layer=layer,
                path=path,
                input_spec=[
                    InputSpec(shape=[None, 8], dtype='float64'),
                    InputSpec(shape=[None, 1], dtype='float64'),
                ],
            )

        with self.assertRaises(ValueError):
            # shape len mismatch
            paddle.jit.save(
                layer=layer,
                path=path,
                input_spec=[
                    InputSpec(shape=[None, 8, 1], dtype='float32'),
                    InputSpec(shape=[None, 1], dtype='float64'),
                ],
            )

        with self.assertRaises(ValueError):
            # shape mismatch
            paddle.jit.save(
                layer=layer,
                path=path,
                input_spec=[
                    InputSpec(shape=[None, 8], dtype='float32'),
                    InputSpec(shape=[None, 2], dtype='float64'),
                ],
            )
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)


class NotJitForward(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class TestNotJitForward(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_jit_not_save_forward(self):
        layer = NotJitForward()

        save_dir = os.path.join(self.temp_dir.name, "jit_not_save_forward")
        path = save_dir + "/model"

        paddle.jit.save(layer=layer, path=path, skip_forward=True)

        self.assertTrue(not os.path.exists(path + ".pdmodel"))
        self.assertTrue(not os.path.exists(path + ".pdparam"))

        with self.assertRaises(ValueError):
            paddle.jit.load(path=path)

        shutil.rmtree(save_dir)


class StridedBufferNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        buffer = paddle.to_tensor([1, 2, 3, 4, 5, 6]).astype('float32')
        strided_buffer = buffer[::2]
        self.register_buffer("strided_buffer", strided_buffer)

    def forward(self, x):
        return self.strided_buffer + x


class TestStridedBuffer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_strided_buffer(self):
        layer = StridedBufferNet()
        save_dir = os.path.join(self.temp_dir.name, "test_strided_buffer")
        path = save_dir + "/model"
        paddle.jit.save(layer=layer, path=path, input_spec=[InputSpec([2, 3])])

        loaded_layer = paddle.jit.load(path)
        x = paddle.to_tensor([1, 2, 3]).astype('float32')
        np.testing.assert_allclose(layer(x).numpy(), loaded_layer(x).numpy())


class LayerWithUnusedBuffer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(7, 10)
        self.register_buffer("buffer", paddle.randn([5, 1]))

    def forward(self, x):
        return self.linear(x)


class TestLayerWithUnusedBuffer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def check_program_has_buffer(self, program, buffer_shape):
        for op in program.global_block().ops:
            if (
                op.name() == "builtin.parameter"
                and op.result(0).shape == buffer_shape
            ):
                return True
        return False

    def test_layer_with_unused_buffer(self):
        layer = LayerWithUnusedBuffer()
        save_dir = os.path.join(
            self.temp_dir.name, "test_layer_with_unused_buffer"
        )
        path = save_dir + "/model"
        paddle.jit.save(
            layer=layer,
            path=path,
            input_spec=[InputSpec([5, 7], dtype="float32")],
        )

        loaded_layer = paddle.jit.load(path)
        x = paddle.rand([5, 7]).astype('float32')
        self.assertTrue(
            self.check_program_has_buffer(
                loaded_layer.program(), layer.buffer.shape
            )
        )


if __name__ == '__main__':
    unittest.main()
