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

__all__ = ['MKLDNNPostTrainingQuantStrategy']

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class MKLDNNPostTrainingQuantStrategy(Strategy):
    """
    The strategy for Post Training quantization strategy.
    """

    def __init__(self, int8_model_save_path=None, fp32_model_path=None):
        """
        Args:
            int8_model_save_path(str): The path to save int8 model with fp32 weights.
                            None means it doesn't save int8 model. defalut: None.
            fp32_model_path(str): The path to model with fp32 weight. defalut: None.
        """

        super(MKLDNNPostTrainingQuantStrategy, self).__init__(0, 0)
        self.int8_model_save_path = int8_model_save_path
        if fp32_model_path is None:
            raise Exception("fp32_model_path is None")
        self.fp32_model_path = fp32_model_path
        if 'FLAGS_OMP_NUM_THREADS' in os.environ:
            self.omp_num_threads = int(os.environ['FLAGS_OMP_NUM_THREADS'])
        else:
            self.omp_num_threads = 1

    def on_compression_begin(self, context):
        """
	Prepare the data and quantify the model
        """

        super(MKLDNNPostTrainingQuantStrategy,
              self).on_compression_begin(context)
        _logger.info('InferQuantStrategy::on_compression_begin')

        #Prepare the Analysis Config
        infer_config = core.AnalysisConfig("AnalysisConfig")
        infer_config.switch_ir_optim(True)
        infer_config.disable_gpu()
        infer_config.set_model(self.fp32_model_path)
        infer_config.enable_mkldnn()
        infer_config.set_cpu_math_library_num_threads(self.omp_num_threads)

        #Prepare the data for calculating the quantization scales
        warmup_reader = context.eval_reader()
        if six.PY2:
            data = warmup_reader.next()

        if six.PY3:
            data = warmup_reader.__next__()

        num_images = len(data)
        images = core.PaddleTensor()
        images.name = "x"
        images.shape = [num_images, ] + list(data[0][0].shape)
        images.dtype = core.PaddleDType.FLOAT32
        image_data = [img.tolist() for (img, _) in data]
        image_data = np.array(image_data).astype("float32")
        image_data = image_data.ravel()
        images.data = core.PaddleBuf(image_data.tolist())

        labels = core.PaddleTensor()
        labels.name = "y"
        labels.shape = [num_images, 1]
        labels.dtype = core.PaddleDType.INT64
        label_data = [label for (_, label) in data]
        labels.data = core.PaddleBuf(label_data)

        warmup_data = [images, labels]

        #Enable the int8 quantization
        infer_config.enable_quantizer()
        infer_config.quantizer_config().set_quant_data(warmup_data)
        infer_config.quantizer_config().set_quant_batch_size(num_images)

        #Run INT8 MKL-DNN Quantization
        predictor = core.create_paddle_predictor(infer_config)
        if self.int8_model_save_path:
            if not os.path.exists(self.int8_model_save_path):
                os.makedirs(self.int8_model_save_path)
            predictor.SaveOptimModel(self.int8_model_save_path)

        _logger.info(
            'Finish MKLDNNPostTrainingQuantStrategy::on_compresseion_begin')
