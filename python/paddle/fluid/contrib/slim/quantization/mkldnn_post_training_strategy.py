# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import six
import numpy as np
from .... import core
from ..core.strategy import Strategy
from ....log_helper import get_logger

__all__ = ['MKLDNNPostTrainingQuantStrategy']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class MKLDNNPostTrainingQuantStrategy(Strategy):
    """
    The strategy for MKL-DNN Post Training quantization strategy.
    """

    def __init__(self,
                 int8_model_save_path=None,
                 fp32_model_path=None,
                 cpu_math_library_num_threads=1):
        """
        Args:
            int8_model_save_path(str): int8_model_save_path is used to save an int8 ProgramDesc
                        with fp32 weights which is used for MKL-DNN int8 inference. For post training quantization,
                        MKLDNNPostTrainingQuantStrategy only supports converting a fp32 ProgramDesc
                        with fp32 weights to an int8 ProgramDesc with fp32 weights now. The saved
                        int8 ProgramDesc with fp32 weights only can be executed with MKL-DNN enabled.
                        None means it doesn't save int8 ProgramDesc with fp32 weights. default: None.
            fp32_model_path(str): fp32_model_path is used to load an original fp32 ProgramDesc with fp32 weights.
                        None means it doesn't have a fp32 ProgramDesc with fp32 weights. default: None.
            cpu_math_library_num_threads(int): The number of cpu math library threads which is used on
                        MKLDNNPostTrainingQuantStrategy. 1 means it only uses one cpu math library
                        thread. default: 1
        """

        super(MKLDNNPostTrainingQuantStrategy, self).__init__(0, 0)
        self.int8_model_save_path = int8_model_save_path
        if fp32_model_path is None:
            raise Exception("fp32_model_path is None")
        self.fp32_model_path = fp32_model_path
        self.cpu_math_library_num_threads = cpu_math_library_num_threads

    def on_compression_begin(self, context):
        """
        Prepare the data and quantify the model
        """

        super(MKLDNNPostTrainingQuantStrategy,
              self).on_compression_begin(context)
        _logger.info('InferQuantStrategy::on_compression_begin')

        # Prepare the Analysis Config
        infer_config = core.AnalysisConfig("AnalysisConfig")
        infer_config.switch_ir_optim(True)
        infer_config.disable_gpu()
        infer_config.set_model(self.fp32_model_path)
        infer_config.enable_mkldnn()
        infer_config.set_cpu_math_library_num_threads(
            self.cpu_math_library_num_threads)

        # Prepare the data for calculating the quantization scales
        warmup_reader = context.eval_reader()
        if six.PY2:
            data = warmup_reader.next()

        if six.PY3:
            data = warmup_reader.__next__()

        num_images = len(data)
        image_data = [img.tolist() for (img, _) in data]
        image_data = np.array(image_data).astype("float32").reshape(
            [num_images, ] + list(data[0][0].shape))
        image_data = image_data.ravel()
        images = core.PaddleTensor(image_data, "x")
        images.shape = [num_images, ] + list(data[0][0].shape)

        label_data = [label for (_, label) in data]
        labels = core.PaddleTensor(
            np.array(label_data).astype("int64").reshape([num_images, 1]), "y")

        warmup_data = [images, labels]

        # Enable the INT8 Quantization
        infer_config.enable_quantizer()
        infer_config.quantizer_config().set_quant_data(warmup_data)
        infer_config.quantizer_config().set_quant_batch_size(num_images)

        # Run INT8 MKL-DNN Quantization
        predictor = core.create_paddle_predictor(infer_config)
        if self.int8_model_save_path:
            if not os.path.exists(self.int8_model_save_path):
                os.makedirs(self.int8_model_save_path)
            predictor.SaveOptimModel(self.int8_model_save_path)

        _logger.info(
            'Finish MKLDNNPostTrainingQuantStrategy::on_compresseion_begin')
