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
import unittest
import os
import numpy as np
import time
import sys
import random
import paddle
import paddle.fluid as fluid
import argparse
import functools
import contextlib
import paddle.fluid.profiler as profiler
from paddle.dataset.common import download
from PIL import Image, ImageEnhance
import math
sys.path.append('..')
import int8_inference.utility as int8_utility

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 1
BUF_SIZE = 102400

DATA_DIR = 'data/ILSVRC2012'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


# TODO(guomingz): Remove duplicated code from line 45 ~ line 114
def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(sample, mode, color_jitter, rotate):
    img_path = sample[0]

    img = Image.open(img_path)

    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    return img, sample[1]


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR):
    def reader():
        with open(file_list) as flist:
            full_lines = [line.strip() for line in flist]
            if shuffle:
                np.random.shuffle(full_lines)

            lines = full_lines

            for line in lines:
                img_path, label = line.split()
                img_path = os.path.join(data_dir, img_path)
                if not os.path.exists(img_path):
                    continue
                yield img_path, int(label)

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def val(data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val_list.txt')
    return _reader_creator(file_list, 'val', shuffle=False, data_dir=data_dir)


class TestCalibrationForResnet50(unittest.TestCase):
    def setUp(self):
        self.int8_download = 'int8/download'
        self.cache_folder = os.path.expanduser('~/.cache/paddle/dataset/' +
                                               self.int8_download)

        data_url = 'http://paddle-inference-dist.cdn.bcebos.com/int8/calibration_test_data.tar.gz'
        data_md5 = '1b6c1c434172cca1bf9ba1e4d7a3157d'
        self.data_cache_folder = self.download_data(data_url, data_md5, "data")

        # reader/decorator.py requires the relative path to the data folder
        cmd = 'rm -rf {0} && ln -s {1} {0}'.format("data",
                                                   self.data_cache_folder)
        os.system(cmd)

        self.iterations = 50

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = 'mkdir {0} && tar xf {1} -C {0}'.format(target_folder,
                                                          zip_path)
            os.system(cmd)

    def download_data(self, data_url, data_md5, folder_name):
        download(data_url, self.int8_download, data_md5)
        data_cache_folder = os.path.join(self.cache_folder, folder_name)
        file_name = data_url.split('/')[-1]
        zip_path = os.path.join(self.cache_folder, file_name)
        self.cache_unzipping(data_cache_folder, zip_path)
        return data_cache_folder

    def download_resnet50_model(self):
        # resnet50 fp32 data
        data_url = 'http://paddle-inference-dist.cdn.bcebos.com/int8/resnet50_int8_model.tar.gz'
        data_md5 = '4a5194524823d9b76da6e738e1367881'
        self.model_cache_folder = self.download_data(data_url, data_md5,
                                                     "resnet50_fp32")

    def run_program(self, model_path, generate_int8=False, algo='direct'):
        image_shape = [3, 224, 224]
        os.environ['FLAGS_use_mkldnn'] = 'True'

        fluid.memory_optimize(fluid.default_main_program())

        exe = fluid.Executor(fluid.CPUPlace())

        [infer_program, feed_dict,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)

        t = fluid.transpiler.InferenceTranspiler()
        t.transpile(infer_program, fluid.CPUPlace())

        val_reader = paddle.batch(val(), batch_size=1)

        if generate_int8:
            int8_model = os.path.join(os.getcwd(), "calibration_out")

            if os.path.exists(int8_model):
                os.system("rm -rf " + int8_model)
                os.system("mkdir " + int8_model)

            print("Start calibration ...")

            calibrator = int8_utility.Calibrator(
                program=infer_program,
                pretrained_model=model_path,
                algo=algo,
                exe=exe,
                output=int8_model,
                feed_var_names=feed_dict,
                fetch_list=fetch_targets)

        test_info = []
        cnt = 0
        for batch_id, data in enumerate(val_reader()):
            image = np.array(
                [x[0].reshape(image_shape) for x in data]).astype("float32")
            label = np.array([x[1] for x in data]).astype("int64")
            label = label.reshape([-1, 1])
            running_program = calibrator.sampling_program.clone(
            ) if generate_int8 else infer_program.clone()
            for op in running_program.current_block().ops:
                if op.has_attr("use_mkldnn"):
                    op._set_attr("use_mkldnn", True)

            _, acc1, _ = exe.run(
                running_program,
                feed={feed_dict[0]: image,
                      feed_dict[1]: label},
                fetch_list=fetch_targets)
            if generate_int8:
                calibrator.sample_data()

            test_info.append(np.mean(acc1) * len(data))
            cnt += len(data)

            if batch_id != self.iterations - 1:
                continue

            break

        if generate_int8:
            calibrator.save_int8_model()

            print(
                "Calibration is done and the corresponding files are generated at {}".
                format(os.path.abspath("calibration_out")))
        else:
            return np.sum(test_info) / cnt

    def test_calibration(self):
        self.download_resnet50_model()
        fp32_acc1 = self.run_program(self.model_cache_folder + "/model")
        self.run_program(self.model_cache_folder + "/model", True)
        int8_acc1 = self.run_program("calibration_out")
        delta_value = np.abs(fp32_acc1 - int8_acc1)
        self.assertLess(delta_value, 0.01)


class TestCalibrationForMobilenetv1(TestCalibrationForResnet50):
    def download_mobilenetv1_model(self):
        # mobilenetv1 fp32 data
        data_url = 'http://paddle-inference-dist.cdn.bcebos.com/int8/mobilenetv1_int8_model.tar.gz'
        data_md5 = '13892b0716d26443a8cdea15b3c6438b'
        self.model_cache_folder = self.download_data(data_url, data_md5,
                                                     "mobilenetv1_fp32")

    def test_calibration(self):
        self.download_mobilenetv1_model()
        fp32_acc1 = self.run_program(self.model_cache_folder + "/model")
        self.run_program(self.model_cache_folder + "/model", True, algo='KL')
        int8_acc1 = self.run_program("calibration_out")
        delta_value = np.abs(fp32_acc1 - int8_acc1)
        self.assertLess(delta_value, 0.01)


if __name__ == '__main__':
    unittest.main()
