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
import numpy as np
import paddle.fluid as fluid
from mobilenet import MobileNet
from paddle.fluid.contrib.slim.core import Compressor
from paddle.fluid.contrib.slim.graph import GraphWrapper
sys.path.append('../../tests')
from test_calibration_resnet50 import TestCalibration


class TestInferQuantizeStrategy(TestCalibration):
    """
    Test API of Post Training quantization strategy for int8 with MKLDNN.
    """

    def download(self):
        self.download_data([
            'http://paddle-inference-dist.bj.bcebos.com/int8/mobilenetv1_int8_model.tar.gz'
        ], ['13892b0716d26443a8cdea15b3c6438b'], 'mobilenetv1_int8_model')
        self.download_data([
            'http://paddle-inference-dist.bj.bcebos.com/int8/imagenet_val_100_tail.tar.gz'
        ], ['b6e05365252f12f75e7daf3fcbc62e96'], 'imagenet_val_100_tail', False)

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

    def test_infer_quant(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return
        self.download()
        config_path = './quantization/config_mkldnn_int8.yaml'
        with open(config_path, 'r+') as fp:
            data = fp.read()
        model_path = os.path.join(self.cache_folder,
                                  'mobilenetv1_int8_model/model')
        data = data.replace('MODEL_PATH', model_path)
        with open(config_path, 'w') as fp:
            fp.write(data)

        data_path = os.path.join(self.cache_folder,
                                 'imagenet_val_100_tail/data.bin')
        #warmup dataset, only use the first batch data
        test_reader = paddle.batch(
            self._reader_creator(data_path, False), batch_size=100)
        com_pass = Compressor(
            None,
            None,
            None,
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


if __name__ == '__main__':
    unittest.main()
