#   copyright (c) 2022 paddlepaddle authors. all rights reserved.
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
import tempfile
import time
import unittest

import numpy as np

import paddle
from paddle.dataset.common import md5file
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
        return {"img": batch}

    def __len__(self):
        return len(self.mnist_data)


class TestPostTrainingQuantization(unittest.TestCase):
    def setUp(self):
        self.root_path = tempfile.TemporaryDirectory()
        self.int8_model_path = os.path.join(
            self.root_path.name, "post_training_quantization"
        )
        self.download_path = f'download_model_{time.time()}'
        self.cache_folder = os.path.join(
            self.root_path.name, self.download_path
        )
        try:
            os.system("mkdir -p " + self.int8_model_path)
            os.system("mkdir -p " + self.cache_folder)
        except Exception as e:
            print(f"Failed to create {self.int8_model_path} due to {e}")
            sys.exit(-1)

    def tearDown(self):
        self.root_path.cleanup()

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = (
                f'mkdir {target_folder} && tar xf {zip_path} -C {target_folder}'
            )
            os.system(cmd)

    def download(self, url, dirname, md5sum, save_name=None):
        import shutil

        import httpx

        filename = os.path.join(
            dirname, url.split('/')[-1] if save_name is None else save_name
        )

        if os.path.exists(filename) and md5file(filename) == md5sum:
            return filename

        retry = 0
        retry_limit = 3
        while not (os.path.exists(filename) and md5file(filename) == md5sum):
            if os.path.exists(filename):
                sys.stderr.write(f"file {md5file(filename)}  md5 {md5sum}\n")
            if retry < retry_limit:
                retry += 1
            else:
                raise RuntimeError(
                    f"Cannot download {url} within retry limit {retry_limit}"
                )
            sys.stderr.write(
                f"Cache file {filename} not found, downloading {url} \n"
            )
            sys.stderr.write("Begin to download\n")
            try:
                with httpx.stream("GET", url) as r:
                    total_length = r.headers.get('content-length')

                    if total_length is None:
                        with open(filename, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
                    else:
                        with open(filename, 'wb') as f:
                            chunk_size = 4096
                            total_length = int(total_length)
                            total_iter = total_length / chunk_size + 1
                            log_interval = (
                                total_iter // 20 if total_iter > 20 else 1
                            )
                            log_index = 0
                            bar = paddle.hapi.progressbar.ProgressBar(
                                total_iter, name='item'
                            )
                            for data in r.iter_bytes(chunk_size=chunk_size):
                                f.write(data)
                                log_index += 1
                                bar.update(log_index, {})
                                if log_index % log_interval == 0:
                                    bar.update(log_index)

            except Exception as e:
                # re-try
                continue
        sys.stderr.write("\nDownload finished\n")
        sys.stdout.flush()
        return filename

    def download_model(self, data_url, data_md5, folder_name):
        self.download(data_url, self.cache_folder, data_md5)
        os.system(f'wget -q {data_url}')
        file_name = data_url.split('/')[-1]
        zip_path = os.path.join(self.cache_folder, file_name)
        print(
            f'Data is downloaded at {zip_path}. File exists: {os.path.exists(zip_path)}'
        )

        data_cache_folder = os.path.join(self.cache_folder, folder_name)
        self.cache_unzipping(data_cache_folder, zip_path)
        return data_cache_folder

    def run_program(
        self,
        model_path,
        model_filename,
        params_filename,
        batch_size,
        infer_iterations,
    ):
        print(
            f"test model path: {model_path}. File exists: {os.path.exists(model_path)}"
        )
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        [
            infer_program,
            feed_dict,
            fetch_targets,
        ] = paddle.static.load_inference_model(
            model_path,
            exe,
            model_filename=model_filename,
            params_filename=params_filename,
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
        model_filename,
        params_filename,
        algo="KL",
        round_type="round",
        quantizable_op_type=["conv2d"],
        is_full_quantize=False,
        is_use_cache_file=False,
        is_optimize_model=False,
        batch_size=10,
        batch_nums=10,
        onnx_format=False,
        skip_tensor_list=None,
        bias_correction=False,
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
            model_filename=model_filename,
            params_filename=params_filename,
            sample_generator=None,
            data_loader=val_data_generator,
            batch_size=batch_size,
            batch_nums=batch_nums,
            algo=algo,
            quantizable_op_type=quantizable_op_type,
            round_type=round_type,
            is_full_quantize=is_full_quantize,
            optimize_model=is_optimize_model,
            bias_correction=bias_correction,
            onnx_format=onnx_format,
            skip_tensor_list=skip_tensor_list,
            is_use_cache_file=is_use_cache_file,
        )
        ptq.quantize()
        ptq.save_quantized_model(self.int8_model_path)

    def run_test(
        self,
        model_name,
        model_filename,
        params_filename,
        data_url,
        data_md5,
        algo,
        round_type,
        quantizable_op_type,
        is_full_quantize,
        is_use_cache_file,
        is_optimize_model,
        diff_threshold,
        batch_size=10,
        infer_iterations=10,
        quant_iterations=5,
        bias_correction=False,
        onnx_format=False,
        skip_tensor_list=None,
    ):
        origin_model_path = self.download_model(data_url, data_md5, model_name)
        origin_model_path = os.path.join(origin_model_path, model_name)

        print(
            f"Start FP32 inference for {model_name} on {infer_iterations * batch_size} images ..."
        )

        (fp32_throughput, fp32_latency, fp32_acc1) = self.run_program(
            origin_model_path,
            model_filename,
            params_filename,
            batch_size,
            infer_iterations,
        )

        print(
            f"Start INT8 post training quantization for {model_name} on {quant_iterations * batch_size} images ..."
        )
        self.generate_quantized_model(
            origin_model_path,
            model_filename,
            params_filename,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            batch_size,
            quant_iterations,
            onnx_format,
            skip_tensor_list,
            bias_correction,
        )

        print(
            f"Start INT8 inference for {model_name} on {infer_iterations * batch_size} images ..."
        )
        (int8_throughput, int8_latency, int8_acc1) = self.run_program(
            self.int8_model_path,
            'model.pdmodel',
            'model.pdiparams',
            batch_size,
            infer_iterations,
        )

        print(f"---Post training quantization of {algo} method---")
        print(
            f"FP32 {model_name}: batch_size {batch_size}, throughput {fp32_throughput} img/s, latency {fp32_latency} s, acc1 {fp32_acc1}."
        )
        print(
            f"INT8 {model_name}: batch_size {batch_size}, throughput {int8_throughput} img/s, latency {int8_latency} s, acc1 {int8_acc1}.\n"
        )
        sys.stdout.flush()

        delta_value = fp32_acc1 - int8_acc1
        self.assertLess(delta_value, diff_threshold)


class TestPostTrainingKLForMnist(TestPostTrainingQuantization):
    def test_post_training_kl(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "KL"
        round_type = "round"
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
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTraininghistForMnist(TestPostTrainingQuantization):
    def test_post_training_hist(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "hist"
        round_type = "round"
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
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingmseForMnist(TestPostTrainingQuantization):
    def test_post_training_mse(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "mse"
        round_type = "round"
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
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingemdForMnist(TestPostTrainingQuantization):
    def test_post_training_mse(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "emd"
        round_type = "round"
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
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingavgForMnist(TestPostTrainingQuantization):
    def test_post_training_avg(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "avg"
        round_type = "round"
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
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingAbsMaxForMnist(TestPostTrainingQuantization):
    def test_post_training_abs_max(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "abs_max"
        round_type = "round"
        quantizable_op_type = ["conv2d", "mul"]
        is_full_quantize = True
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 10
        self.run_test(
            model_name,
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingmseAdaroundForMnist(TestPostTrainingQuantization):
    def test_post_training_mse(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "mse"
        round_type = "adaround"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        bias_correction = True
        self.run_test(
            model_name,
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
            bias_correction=bias_correction,
        )


class TestPostTrainingKLAdaroundForMnist(TestPostTrainingQuantization):
    def test_post_training_kl(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "KL"
        round_type = "adaround"
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
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
        )


class TestPostTrainingmseForMnistONNXFormat(TestPostTrainingQuantization):
    def test_post_training_mse_onnx_format(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "mse"
        round_type = "round"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        onnx_format = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
            onnx_format=onnx_format,
        )


class TestPostTrainingmseForMnistONNXFormatFullQuant(
    TestPostTrainingQuantization
):
    def test_post_training_mse_onnx_format_full_quant(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "mse"
        round_type = "round"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = True
        is_use_cache_file = False
        is_optimize_model = False
        onnx_format = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
            onnx_format=onnx_format,
        )


class TestPostTrainingavgForMnistSkipOP(TestPostTrainingQuantization):
    def test_post_training_avg_skip_op(self):
        model_name = "mnist_model"
        data_url = "http://paddle-inference-dist.bj.bcebos.com/int8/mnist_model_combined.tar.gz"
        data_md5 = "a49251d3f555695473941e5a725c6014"
        algo = "avg"
        round_type = "round"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = False
        is_use_cache_file = False
        is_optimize_model = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        skip_tensor_list = ["fc_0.w_0"]
        self.run_test(
            model_name,
            'model.pdmodel',
            'model.pdiparams',
            data_url,
            data_md5,
            algo,
            round_type,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
            skip_tensor_list=skip_tensor_list,
        )


if __name__ == '__main__':
    unittest.main()
