# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import six
import numpy as np
import platform
import os
from .... import Executor
from .... import io
from .... import core
from ....compiler import CompiledProgram
from ....framework import IrGraph
from ..core.strategy import Strategy
import logging
import sys

__all__ = ['InferQuantStrategy']

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class InferQuantStrategy(Strategy):
    """
    The strategy for Post Training quantization strategy.
    """

    def __init__(self,
                 start_epoch=0,
                 end_epoch=0,
                 int8_model_save_path=None,
                 fp32_model_path=None):
        """
        Args:
            start_epoch(int): The 'on_epoch_begin' function will be called in start_epoch. default: 0
            end_epoch(int): The 'on_epoch_end' function will be called in end_epoch. default: 0
            int8_model_save_path(str): The path to save model with int8_t weight.
                            None means it doesn't save int8 model. defalut: None.
            fp32_model_path(str): The path to model with fp32 weight. defalut: None.

        """

        super(InferQuantStrategy, self).__init__(start_epoch, end_epoch)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.int8_model_save_path = int8_model_save_path
        self.fp32_model_path = fp32_model_path

    def on_compression_begin(self, context):
        """
	Prepare the data and quantify the model
        """
        super(InferQuantStrategy, self).on_compression_begin(context)
        if context.epoch_id == self.end_epoch:
            logger.info('InferQuantStrategy::on_compression_begin')

            #Prepare the Analysis Config
            infer_config = core.AnalysisConfig("AnalysisConfig")
            infer_config.switch_ir_optim(True)
            infer_config.disable_gpu
            infer_config.set_model(self.fp32_model_path)
            infer_config.enable_mkldnn()

            #Prepare the data for calculating the quantization scales 
            warmup_reader = context.eval_reader()
            if six.PY2:
                data = warmup_reader.next()

            if six.PY3:
                data = warmup_reader.__next__()

            num_images = len(data)
            dshape = [3, 224, 224]
            shape = [num_images, 3, 224, 224]
            image = core.PaddleTensor()
            image.name = "x"
            image.shape = shape
            image.dtype = core.PaddleDType.FLOAT32
            image.data.resize(num_images * 3 * 224 * 224 * sys.getsizeof(float))
            if six.PY2:
                image_data = map(lambda x: x[0].reshape(dshape), data)
            if six.PY3:
                image_data = list(map(lambda x: x[0].reshape(dshape), data))
            image_data = np.array(image_data).astype("float32")
            image_data = np.reshape(image_data, (np.product(image_data.shape)))
            image.data = core.PaddleBuf(image_data.tolist())

            label = core.PaddleTensor()
            label.name = "y"
            label.shape = [num_images, 1]
            label.dtype = core.PaddleDType.INT64
            image.data.resize(num_images * sys.getsizeof(int))
            if six.PY2:
                label_data = map(lambda x: x[1], data)
            if six.PY3:
                label_data = list(map(lambda x: x[1], data))
            label_data = np.array(label_data)
            label_data = label_data.reshape([-1, 1])
            label.data = core.PaddleBuf(label_data)

            warmup_data = [image, label]

            #Enable the int8 quantization
            infer_config.enable_quantizer()
            infer_config.quantizer_config().set_quant_data(warmup_data)
            infer_config.quantizer_config().set_quant_batch_size(num_images)
            #Run INT8 MKL-DNN Quantization
            predictor = core.create_paddle_predictor(infer_config)
            if not os.path.exists(self.int8_model_save_path):
                os.makedirs(self.int8_model_save_path)
            predictor.SaveOptimModel(self.int8_model_save_path)

            logger.info('Finish InferQuantStrategy::on_compresseion_begin')
