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

import os
import tempfile
import unittest

import numpy as np

import paddle
import paddle.optimizer as opt
from paddle import nn

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4
SEED = 10

IMAGE_SIZE = 784
CLASS_NUM = 10


# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        np.random.seed(SEED)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
        self._dropout = paddle.nn.Dropout(p=0.5)

    @paddle.jit.to_static(
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, IMAGE_SIZE], dtype='float32', name='x'
            )
        ],
        full_graph=True,
    )
    def forward(self, x):
        return self._linear(x)


def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            print(
                f"Epoch {epoch_id} batch {batch_id}: loss = {np.mean(loss.numpy())}"
            )
    return loss


class TestTranslatedLayer(unittest.TestCase):
    def tearDown(self):
        self.temp_dir.cleanup()

    def setUp(self):
        # enable dygraph mode
        place = paddle.CPUPlace()
        paddle.disable_static(place)

        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        # create network
        self.layer = LinearNet()
        self.loss_fn = nn.CrossEntropyLoss()
        self.sgd = opt.SGD(
            learning_rate=0.001, parameters=self.layer.parameters()
        )

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        self.loader = paddle.io.DataLoader(
            dataset,
            places=place,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        self.temp_dir = tempfile.TemporaryDirectory()

        # train
        train(self.layer, self.loader, self.loss_fn, self.sgd)

        # save
        self.model_path = os.path.join(
            self.temp_dir.name, './linear.example.model'
        )
        paddle.jit.save(self.layer, self.model_path)

    def test_inference_and_fine_tuning(self):
        self.load_and_inference()
        self.load_and_fine_tuning()

    def load_and_inference(self):
        # load
        translated_layer = paddle.jit.load(self.model_path)

        # inference
        x = paddle.randn([1, IMAGE_SIZE], 'float32')

        self.layer.eval()
        orig_pred = self.layer(x)

        translated_layer.eval()
        pred = translated_layer(x)

        np.testing.assert_array_equal(orig_pred.numpy(), pred.numpy())

    def load_and_fine_tuning(self):
        # load
        translated_layer = paddle.jit.load(self.model_path)

        # train original layer continue
        self.layer.train()
        orig_loss = train(self.layer, self.loader, self.loss_fn, self.sgd)

        # fine-tuning
        translated_layer.train()
        sgd = opt.SGD(
            learning_rate=0.001, parameters=translated_layer.parameters()
        )
        loss = train(translated_layer, self.loader, self.loss_fn, sgd)

        np.testing.assert_array_equal(
            orig_loss.numpy(),
            loss.numpy(),
            err_msg=f'original loss:\n{orig_loss.numpy()}\nnew loss:\n{loss.numpy()}\n',
        )

    def test_get_program(self):
        # load
        translated_layer = paddle.jit.load(self.model_path)

        program = translated_layer.program()
        self.assertTrue(isinstance(program, paddle.static.Program))

    def test_get_program_method_not_exists(self):
        # load
        translated_layer = paddle.jit.load(self.model_path)

        with self.assertRaises(ValueError):
            program = translated_layer.program('not_exists')

    def test_get_input_spec(self):
        # load
        translated_layer = paddle.jit.load(self.model_path)

        expect_spec = [
            paddle.static.InputSpec(
                shape=[None, IMAGE_SIZE], dtype='float32', name='x'
            )
        ]
        actual_spec = translated_layer._input_spec()

        for spec_x, spec_y in zip(expect_spec, actual_spec):
            self.assertEqual(spec_x, spec_y)

    def test_get_output_spec(self):
        # load
        translated_layer = paddle.jit.load(self.model_path)

        expect_spec = [
            paddle.static.InputSpec(
                shape=[None, CLASS_NUM],
                dtype='float32',
                name='output_0',
            )
        ]
        actual_spec = translated_layer._output_spec()

        for spec_x, spec_y in zip(expect_spec, actual_spec):
            self.assertEqual(spec_x, spec_y)

    def test_layer_state(self):
        # load
        translated_layer = paddle.jit.load(self.model_path)
        translated_layer.eval()
        self.assertEqual(translated_layer.training, False)
        for layer in translated_layer.sublayers():
            print("123")
            self.assertEqual(layer.training, False)

        translated_layer.train()
        self.assertEqual(translated_layer.training, True)
        for layer in translated_layer.sublayers():
            self.assertEqual(layer.training, True)


if __name__ == '__main__':
    unittest.main()
