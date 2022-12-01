#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import numpy as np
import random
import shutil
import time
import unittest
import copy
import logging
import tempfile

import paddle.nn as nn
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.slim.quantization import *
from paddle.fluid.log_helper import get_logger
from paddle.dataset.common import download
from paddle.fluid.framework import _test_eager_guard

from imperative_test_utils import (
    fix_model_dict,
    ImperativeLenet,
    ImperativeLinearBn,
)
from imperative_test_utils import ImperativeLinearBn_hook

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class TestFuseLinearBn(unittest.TestCase):
    """
    Fuse the linear and bn layers, and then quantize the model.
    """

    def test_fuse(self):
        model = ImperativeLinearBn()
        model_h = ImperativeLinearBn_hook()
        inputs = paddle.randn((3, 10), dtype="float32")
        config = PTQConfig(AbsmaxQuantizer(), AbsmaxQuantizer())
        ptq = ImperativePTQ(config)
        f_l = [['linear', 'bn']]
        quant_model = ptq.quantize(model, fuse=True, fuse_list=f_l)
        quant_h = ptq.quantize(model_h, fuse=True, fuse_list=f_l)
        for name, layer in quant_model.named_sublayers():
            if name in f_l:
                assert not (
                    isinstance(layer, nn.BatchNorm1D)
                    or isinstance(layer, nn.BatchNorm2D)
                )
        out = model(inputs)
        out_h = model_h(inputs)
        out_quant = quant_model(inputs)
        out_quant_h = quant_h(inputs)
        cos_sim_func = nn.CosineSimilarity(axis=0)
        print(
            'fuse linear+bn', cos_sim_func(out.flatten(), out_quant.flatten())
        )
        print(cos_sim_func(out_h.flatten(), out_quant_h.flatten()))


class TestImperativePTQ(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(cls):
        cls.download_path = 'dygraph_int8/download'
        cls.cache_folder = os.path.expanduser(
            '~/.cache/paddle/dataset/' + cls.download_path
        )

        cls.lenet_url = "https://paddle-inference-dist.cdn.bcebos.com/int8/unittest_model_data/lenet_pretrained.tar.gz"
        cls.lenet_md5 = "953b802fb73b52fae42896e3c24f0afb"

        seed = 1
        np.random.seed(seed)
        paddle.static.default_main_program().random_seed = seed
        paddle.static.default_startup_program().random_seed = seed

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = 'mkdir {0} && tar xf {1} -C {0}'.format(
                target_folder, zip_path
            )
            os.system(cmd)

    def download_model(self, data_url, data_md5, folder_name):
        download(data_url, self.download_path, data_md5)
        file_name = data_url.split('/')[-1]
        zip_path = os.path.join(self.cache_folder, file_name)
        print('Data is downloaded at {0}'.format(zip_path))

        data_cache_folder = os.path.join(self.cache_folder, folder_name)
        self.cache_unzipping(data_cache_folder, zip_path)
        return data_cache_folder

    def set_vars(self):
        config = PTQConfig(AbsmaxQuantizer(), AbsmaxQuantizer())
        self.ptq = ImperativePTQ(config)

        self.batch_num = 10
        self.batch_size = 10
        self.eval_acc_top1 = 0.95

        # the input, output and weight thresholds of quantized op
        self.gt_thresholds = {
            'conv2d_0': [[1.0], [0.37673383951187134], [0.10933732241392136]],
            'batch_norm2d_0': [[0.37673383951187134], [0.44249194860458374]],
            're_lu_0': [[0.44249194860458374], [0.25804123282432556]],
            'max_pool2d_0': [[0.25804123282432556], [0.25804123282432556]],
            'linear_0': [
                [1.7058950662612915],
                [14.405526161193848],
                [0.4373355209827423],
            ],
            'add_0': [[1.7058950662612915, 0.0], [1.7058950662612915]],
        }

    def model_test(self, model, batch_num=-1, batch_size=8):
        model.eval()

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )

        eval_acc_top1_list = []
        for batch_id, data in enumerate(test_reader()):
            x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype(
                'float32'
            )
            y_data = (
                np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
            )

            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            out = model(img)
            acc_top1 = paddle.static.accuracy(input=out, label=label, k=1)
            acc_top5 = paddle.static.accuracy(input=out, label=label, k=5)
            eval_acc_top1_list.append(float(acc_top1.numpy()))

            if batch_id % 50 == 0:
                _logger.info(
                    "Test | At step {}: acc1 = {:}, acc5 = {:}".format(
                        batch_id, acc_top1.numpy(), acc_top5.numpy()
                    )
                )

            if batch_num > 0 and batch_id + 1 >= batch_num:
                break

        eval_acc_top1 = sum(eval_acc_top1_list) / len(eval_acc_top1_list)

        return eval_acc_top1

    def program_test(self, program_path, batch_num=-1, batch_size=8):
        exe = paddle.static.Executor(paddle.CPUPlace())
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.load_inference_model(program_path, exe)

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )

        top1_correct_num = 0.0
        total_num = 0.0
        for batch_id, data in enumerate(test_reader()):
            img = np.array([x[0].reshape(1, 28, 28) for x in data]).astype(
                'float32'
            )
            label = np.array([x[1] for x in data]).astype('int64')

            feed = {feed_target_names[0]: img}
            results = exe.run(
                inference_program, feed=feed, fetch_list=fetch_targets
            )

            pred = np.argmax(results[0], axis=1)
            top1_correct_num += np.sum(np.equal(pred, label))
            total_num += len(img)

            if total_num % 50 == 49:
                _logger.info(
                    "Test | Test num {}: acc1 = {:}".format(
                        total_num, top1_correct_num / total_num
                    )
                )

            if batch_num > 0 and batch_id + 1 >= batch_num:
                break
        return top1_correct_num / total_num

    def func_ptq(self):
        start_time = time.time()

        self.set_vars()

        # Load model
        params_path = self.download_model(
            self.lenet_url, self.lenet_md5, "lenet"
        )
        params_path += "/lenet_pretrained/lenet.pdparams"

        model = ImperativeLenet()
        model_state_dict = paddle.load(params_path)
        model.set_state_dict(model_state_dict)
        # Quantize, calibrate and save
        quant_model = self.ptq.quantize(model)
        before_acc_top1 = self.model_test(
            quant_model, self.batch_num, self.batch_size
        )

        input_spec = [
            paddle.static.InputSpec(shape=[None, 1, 28, 28], dtype='float32')
        ]
        with tempfile.TemporaryDirectory(prefix="imperative_ptq_") as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            self.ptq.save_quantized_model(
                model=quant_model, path=save_path, input_spec=input_spec
            )
            print('Quantized model saved in {%s}' % save_path)

            after_acc_top1 = self.model_test(
                quant_model, self.batch_num, self.batch_size
            )

            paddle.enable_static()
            infer_acc_top1 = self.program_test(
                save_path, self.batch_num, self.batch_size
            )
            paddle.disable_static()

            # Check
            print('Before converted acc_top1: %s' % before_acc_top1)
            print('After converted acc_top1: %s' % after_acc_top1)
            print('Infer acc_top1: %s' % infer_acc_top1)

            self.assertTrue(
                after_acc_top1 >= self.eval_acc_top1,
                msg="The test acc {%f} is less than {%f}."
                % (after_acc_top1, self.eval_acc_top1),
            )
            self.assertTrue(
                infer_acc_top1 >= after_acc_top1,
                msg='The acc is lower after converting model.',
            )

            end_time = time.time()
            print("total time: %ss \n" % (end_time - start_time))

    def test_ptq(self):
        with _test_eager_guard():
            self.func_ptq()
        self.func_ptq()


class TestImperativePTQfuse(TestImperativePTQ):
    def func_ptq(self):
        start_time = time.time()

        self.set_vars()

        # Load model
        params_path = self.download_model(
            self.lenet_url, self.lenet_md5, "lenet"
        )
        params_path += "/lenet_pretrained/lenet.pdparams"

        model = ImperativeLenet()
        model_state_dict = paddle.load(params_path)
        model.set_state_dict(model_state_dict)
        # Quantize, calibrate and save
        f_l = [['features.0', 'features.1'], ['features.4', 'features.5']]
        quant_model = self.ptq.quantize(model, fuse=True, fuse_list=f_l)
        for name, layer in quant_model.named_sublayers():
            if name in f_l:
                assert not (
                    isinstance(layer, nn.BatchNorm1D)
                    or isinstance(layer, nn.BatchNorm2D)
                )
        before_acc_top1 = self.model_test(
            quant_model, self.batch_num, self.batch_size
        )

        input_spec = [
            paddle.static.InputSpec(shape=[None, 1, 28, 28], dtype='float32')
        ]
        with tempfile.TemporaryDirectory(prefix="imperative_ptq_") as tmpdir:
            save_path = os.path.join(tmpdir, "model")
            self.ptq.save_quantized_model(
                model=quant_model, path=save_path, input_spec=input_spec
            )
            print('Quantized model saved in {%s}' % save_path)

            after_acc_top1 = self.model_test(
                quant_model, self.batch_num, self.batch_size
            )

            paddle.enable_static()
            infer_acc_top1 = self.program_test(
                save_path, self.batch_num, self.batch_size
            )
            paddle.disable_static()

            # Check
            print('Before converted acc_top1: %s' % before_acc_top1)
            print('After converted acc_top1: %s' % after_acc_top1)
            print('Infer acc_top1: %s' % infer_acc_top1)

            # Check whether the quant_model is correct after converting.
            # The acc of quantized model should be higher than 0.95.
            self.assertTrue(
                after_acc_top1 >= self.eval_acc_top1,
                msg="The test acc {%f} is less than {%f}."
                % (after_acc_top1, self.eval_acc_top1),
            )
            # Check the saved infer_model.The acc of infer model
            # should not be lower than the one of dygraph model.
            self.assertTrue(
                infer_acc_top1 >= after_acc_top1,
                msg='The acc is lower after converting model.',
            )

            end_time = time.time()
            print("total time: %ss \n" % (end_time - start_time))

    def test_ptq(self):
        with _test_eager_guard():
            self.func_ptq()
        self.func_ptq()


class TestImperativePTQHist(TestImperativePTQ):
    def set_vars(self):
        config = PTQConfig(HistQuantizer(), AbsmaxQuantizer())
        self.ptq = ImperativePTQ(config)

        self.batch_num = 10
        self.batch_size = 10
        self.eval_acc_top1 = 0.98

        self.gt_thresholds = {
            'conv2d_0': [
                [0.99853515625],
                [0.35732391771364225],
                [0.10933732241392136],
            ],
            'batch_norm2d_0': [[0.35732391771364225], [0.4291427868761275]],
            're_lu_0': [[0.4291427868761275], [0.2359918110742001]],
            'max_pool2d_0': [[0.2359918110742001], [0.25665526917146053]],
            'linear_0': [
                [1.7037603475152991],
                [14.395224522473026],
                [0.4373355209827423],
            ],
            'add_0': [[1.7037603475152991, 0.0], [1.7037603475152991]],
        }


class TestImperativePTQKL(TestImperativePTQ):
    def set_vars(self):
        config = PTQConfig(KLQuantizer(), PerChannelAbsmaxQuantizer())
        self.ptq = ImperativePTQ(config)

        self.batch_num = 10
        self.batch_size = 10
        self.eval_acc_top1 = 0.98

        conv2d_1_wt_thresholds = [
            0.18116560578346252,
            0.17079241573810577,
            0.1702047884464264,
            0.179476797580719,
            0.1454375684261322,
            0.22981858253479004,
        ]
        self.gt_thresholds = {
            'conv2d_0': [[0.99267578125], [0.37695913558696836]],
            'conv2d_1': [
                [0.19189296757394914],
                [0.24514256547263358],
                [conv2d_1_wt_thresholds],
            ],
            'batch_norm2d_0': [[0.37695913558696836], [0.27462541429440535]],
            're_lu_0': [[0.27462541429440535], [0.19189296757394914]],
            'max_pool2d_0': [[0.19189296757394914], [0.19189296757394914]],
            'linear_0': [[1.2839322163611087], [8.957185942414352]],
            'add_0': [[1.2839322163611087, 0.0], [1.2839322163611087]],
        }


if __name__ == '__main__':
    unittest.main()
