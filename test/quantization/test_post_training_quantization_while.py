#   copyright (c) 2021 paddlepaddle authors. all rights reserved.
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
import random
import sys
import time
import unittest

import numpy as np

import paddle
from paddle.dataset.common import download
from paddle.static.quantization import PostTrainingQuantization

paddle.enable_static()

random.seed(0)
np.random.seed(0)


class TransedMnistDataSet(paddle.io.Dataset):
    def __init__(self, mnist_data):
        self.mnist_data = mnist_data

    def __getitem__(self, idx):
        img = (
            np.array(self.mnist_data[idx][0])
            .astype('float32')
            .reshape(1, 28, 28)
        )
        batch = img / 127.5 - 1.0
        return {"x": batch}

    def __len__(self):
        return len(self.mnist_data)


class TestPostTrainingQuantization(unittest.TestCase):
    def setUp(self):
        self.download_path = 'int8/download'
        self.cache_folder = os.path.expanduser(
            '~/.cache/paddle/dataset/' + self.download_path
        )
        self.timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.int8_model_path = os.path.join(
            os.getcwd(), "post_training_" + self.timestamp
        )
        try:
            os.system("mkdir -p " + self.int8_model_path)
        except Exception as e:
            print(f"Failed to create {self.int8_model_path} due to {str(e)}")
            sys.exit(-1)

    def tearDown(self):
        try:
            os.system(f"rm -rf {self.int8_model_path}")
        except Exception as e:
            print(f"Failed to delete {self.int8_model_path} due to {str(e)}")

    def cache_unzipping(self, target_folder, zip_path):
        cmd = f'tar xf {zip_path} -C {target_folder}'
        os.system(cmd)

    def download_model(self, data_url, data_md5, folder_name):
        download(data_url, self.download_path, data_md5)
        file_name = data_url.split('/')[-1]
        zip_path = os.path.join(self.cache_folder, file_name)
        print(f'Data is downloaded at {zip_path}')

        data_cache_folder = os.path.join(self.cache_folder, folder_name)
        self.cache_unzipping(self.cache_folder, zip_path)
        return data_cache_folder

    def run_program(self, model_path, batch_size, infer_iterations):
        print("test model path:" + model_path)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        [
            infer_program,
            feed_dict,
            fetch_targets,
        ] = paddle.static.load_inference_model(
            model_path,
            model_filename='model.pdmodel',
            params_filename='model.pdiparams',
            executor=exe,
        )
        val_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size)

        img_shape = [1, 28, 28]
        test_info = []
        cnt = 0
        periods = []
        for batch_id, data in enumerate(val_reader()):
            image = np.array([x[0].reshape(img_shape) for x in data]).astype(
                "float32"
            )
            input_label = np.array([x[1] for x in data]).astype("int64")

            t1 = time.time()
            out = exe.run(
                infer_program,
                feed={feed_dict[0]: image},
                fetch_list=fetch_targets,
            )
            t2 = time.time()
            period = t2 - t1
            periods.append(period)

            out_label = np.argmax(np.array(out[0]), axis=1)
            top1_num = sum(input_label == out_label)
            test_info.append(top1_num)
            cnt += len(data)

            if (batch_id + 1) == infer_iterations:
                break

        throughput = cnt / np.sum(periods)
        latency = np.average(periods)
        acc1 = np.sum(test_info) / cnt
        return (throughput, latency, acc1)

    def generate_quantized_model(
        self,
        model_path,
        algo="KL",
        quantizable_op_type=["conv2d"],
        is_full_quantize=False,
        is_use_cache_file=False,
        is_optimize_model=False,
        batch_size=10,
        batch_nums=10,
    ):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', transform=None
        )
        train_dataset = TransedMnistDataSet(train_dataset)
        BatchSampler = paddle.io.BatchSampler(
            train_dataset, batch_size=batch_size
        )
        val_data_generator = paddle.io.DataLoader(
            train_dataset,
            batch_sampler=BatchSampler,
            places=paddle.static.cpu_places(),
        )

        ptq = PostTrainingQuantization(
            executor=exe,
            model_dir=model_path,
            model_filename='model.pdmodel',
            params_filename='model.pdiparams',
            sample_generator=None,
            data_loader=val_data_generator,
            batch_size=batch_size,
            batch_nums=batch_nums,
            algo=algo,
            quantizable_op_type=quantizable_op_type,
            is_full_quantize=is_full_quantize,
            optimize_model=is_optimize_model,
            is_use_cache_file=is_use_cache_file,
        )
        ptq.quantize()
        ptq.save_quantized_model(
            self.int8_model_path,
            model_filename='model.pdmodel',
            params_filename='model.pdiparams',
        )

    def run_test(
        self,
        model_name,
        data_url,
        data_md5,
        algo,
        quantizable_op_type,
        is_full_quantize,
        is_use_cache_file,
        is_optimize_model,
        diff_threshold,
        batch_size=10,
        infer_iterations=10,
        quant_iterations=5,
    ):
        origin_model_path = self.download_model(data_url, data_md5, model_name)

        print(
            "Start FP32 inference for {} on {} images ...".format(
                model_name, infer_iterations * batch_size
            )
        )
        (fp32_throughput, fp32_latency, fp32_acc1) = self.run_program(
            origin_model_path, batch_size, infer_iterations
        )

        print(
            "Start INT8 post training quantization for {} on {} images ...".format(
                model_name, quant_iterations * batch_size
            )
        )
        self.generate_quantized_model(
            origin_model_path,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            batch_size,
            quant_iterations,
        )

        print(
            "Start INT8 inference for {} on {} images ...".format(
                model_name, infer_iterations * batch_size
            )
        )
        (int8_throughput, int8_latency, int8_acc1) = self.run_program(
            self.int8_model_path, batch_size, infer_iterations
        )

        print(f"---Post training quantization of {algo} method---")
        print(
            "FP32 {}: batch_size {}, throughput {} img/s, latency {} s, acc1 {}.".format(
                model_name, batch_size, fp32_throughput, fp32_latency, fp32_acc1
            )
        )
        print(
            "INT8 {}: batch_size {}, throughput {} img/s, latency {} s, acc1 {}.\n".format(
                model_name, batch_size, int8_throughput, int8_latency, int8_acc1
            )
        )
        sys.stdout.flush()

        delta_value = fp32_acc1 - int8_acc1
        self.assertLess(delta_value, diff_threshold)


class TestPostTrainingKLForWhile(TestPostTrainingQuantization):
    def test_post_training_kl(self):
        model_name = "mnist_while"
        data_url = (
            "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_while.tar.gz"
        )
        data_md5 = "2387390beeb37b51dec041c27b8a681f"
        algo = "KL"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            data_url,
            data_md5,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTraininghistForWhile(TestPostTrainingQuantization):
    def test_post_training_hist(self):
        model_name = "mnist_while"
        data_url = (
            "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_while.tar.gz"
        )
        data_md5 = "2387390beeb37b51dec041c27b8a681f"
        algo = "hist"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            data_url,
            data_md5,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingmseForWhile(TestPostTrainingQuantization):
    def test_post_training_mse(self):
        model_name = "mnist_while"
        data_url = (
            "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_while.tar.gz"
        )
        data_md5 = "2387390beeb37b51dec041c27b8a681f"
        algo = "mse"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            data_url,
            data_md5,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingavgForWhile(TestPostTrainingQuantization):
    def test_post_training_avg(self):
        model_name = "mnist_while"
        data_url = (
            "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_while.tar.gz"
        )
        data_md5 = "2387390beeb37b51dec041c27b8a681f"
        algo = "avg"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            data_url,
            data_md5,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingMinMaxForWhile(TestPostTrainingQuantization):
    def test_post_training_min_max(self):
        model_name = "mnist_while"
        data_url = (
            "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_while.tar.gz"
        )
        data_md5 = "2387390beeb37b51dec041c27b8a681f"
        algo = "min_max"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            data_url,
            data_md5,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingAbsMaxForWhile(TestPostTrainingQuantization):
    def test_post_training_abs_max(self):
        model_name = "mnist_while"
        data_url = (
            "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_while.tar.gz"
        )
        data_md5 = "2387390beeb37b51dec041c27b8a681f"
        algo = "abs_max"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            data_url,
            data_md5,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )
        self.run_test(
            model_name,
            data_url,
            data_md5,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


if __name__ == '__main__':
    unittest.main()
