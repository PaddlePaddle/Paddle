#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
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

import paddle
import unittest
import os
import sys
import struct
import shutil
import numpy as np
import paddle.fluid as fluid
from mobilenet import MobileNet
from paddle.fluid.contrib.slim.core import Compressor
from paddle.fluid.contrib.slim.graph import GraphWrapper
sys.path.append('../../tests')
from test_calibration_resnet50 import TestCalibration


class TestMKLDNNPostTrainingQuantStrategy(TestCalibration):
    """
    Test API of Post Training quantization strategy for int8 with MKLDNN.
    """

    def _reader_creator(self, data_file='data.bin', cycle=False):
        def reader():
            with open(data_file, 'rb') as fp:
                num = fp.read(8)
                num = struct.unpack('q', num)[0]
                imgs_offset = 8
                img_ch = 3
                img_w = 224
                img_h = 224
                img_pixel_size = 4
                img_size = img_ch * img_h * img_w * img_pixel_size
                label_size = 8
                labels_offset = imgs_offset + num * img_size
                step = 0

                while step < num:
                    fp.seek(imgs_offset + img_size * step)
                    img = fp.read(img_size)
                    img = struct.unpack_from('{}f'.format(img_ch * img_w *
                                                          img_h), img)
                    img = np.array(img)
                    img.shape = (img_ch, img_w, img_h)
                    fp.seek(labels_offset + label_size * step)
                    label = fp.read(label_size)
                    label = struct.unpack('q', label)[0]
                    yield img, int(label)
                    step += 1
                    if cycle and step == num:
                        step = 0

        return reader

    def _update_config_file(self, model_name):
        config_path = './quantization/config_mkldnn_int8.yaml'
        new_config_path = './quantization/{0}.yaml'.format(model_name)
        shutil.copy(config_path, new_config_path)

        with open(new_config_path, 'r+') as fp:
            data = fp.read()
        model_path = '{0}/model'.format(model_name)
        model_path = os.path.join(self.cache_folder, model_path)
        data = data.replace('MODEL_PATH', model_path)
        output_path = './{0}/int8'.format(model_name)
        data = data.replace('OUTPUT_PATH', output_path)
        with open(new_config_path, 'w') as fp:
            fp.write(data)

        return new_config_path

    def _test_int8_quant(self, data_path, config_path):
        #warmup dataset, only use the first batch data
        test_reader = paddle.batch(
            self._reader_creator(data_path, False), batch_size=100)
        com_pass = Compressor(
            place=None,
            scope=None,
            train_program=None,
            train_reader=None,
            train_feed_list=[],
            train_fetch_list=[],
            eval_program=None,
            eval_reader=test_reader,
            eval_feed_list=[],
            eval_fetch_list=[],
            teacher_programs=[],
            checkpoint_path='',
            train_optimizer=None,
            distiller_optimizer=None)
        com_pass.config(config_path)
        eval_graph = com_pass.run()

    def test_compression(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        base_url = 'http://paddle-inference-dist.bj.bcebos.com/int8/'
        self.download_data([base_url + 'imagenet_val_100_tail.tar.gz'],
                           ['b6e05365252f12f75e7daf3fcbc62e96'],
                           'imagenet_val_100_tail', False)
        data_path = os.path.join(self.cache_folder,
                                 'imagenet_val_100_tail/data.bin')

        model_urls = [
            base_url + 'mobilenetv1_int8_model.tar.gz',
            base_url + 'resnet50_int8_model.tar.gz'
        ]
        md5s = [
            '13892b0716d26443a8cdea15b3c6438b',
            '4a5194524823d9b76da6e738e1367881'
        ]
        for model_url, md5 in zip(model_urls, md5s):
            model_name = model_url.split('/')[-1]
            model_name = model_name.split('.')[0]
            self.download_data([model_url], [md5], model_name)
            config_path = self._update_config_file(model_name)
            self._test_int8_quant(data_path, config_path)


if __name__ == '__main__':
    unittest.main()
