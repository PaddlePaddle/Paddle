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

import numpy as np
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
    The strategy for Quantization.
    """

    def __init__(self,
                 start_epoch=0,
                 end_epoch=0,
                 int8_model_save_path=None,
                 activation_quantize_type='abs_max'):
        super(InferQuantStrategy, self).__init__(start_epoch, end_epoch)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.int8_model_save_path = int8_model_save_path
        self.activation_quantize_type = activation_quantize_type

    def on_compression_begin(self, context):
        super(InferQuantStrategy, self).on_compression_begin(context)
        if context.epoch_id == self.end_epoch:
            logger.info('InferQuantStrategy::on_compression_begin')
            test_graph = IrGraph(
                core.Graph(context.eval_graph.program.desc), for_test=True)
            build_strategy = core.ParallelExecutor.BuildStrategy()
            scope = context.scope

            #Prepare the Analysis Config
            infer_config = core.AnalysisConfig("AnalysisConfig")
            infer_config.switch_ir_optim(True)
            infer_config.disable_gpu
            infer_config.set_model(context.load_model_dir)
            infer_config.enable_mkldnn()

            #Prepare the data for calculating the quantization scales 
            warmup_data = []
            num_images = 100
            dshape = [3, 224, 224]
            shape = [num_images, 3, 224, 224]
            image = core.PaddleTensor()
            image.name = "x"
            image.shape = shape
            image.dtype = core.PaddleDType.FLOAT32
            image.data.resize(num_images * 3 * 224 * 224 * sys.getsizeof(float))

            label = core.PaddleTensor()
            label.name = "y"
            label.shape = [num_images, 1]
            label.dtype = core.PaddleDType.INT64
            image.data.resize(num_images * sys.getsizeof(int))

            for batch_id, data in enumerate(context.eval_reader()):
                image_data = np.array(
                    map(lambda x: x[0].reshape(dshape), data)).astype("float32")
                image_data = np.reshape(image_data,
                                        (np.product(image_data.shape)))
                label_data = np.array(map(lambda x: x[1], data)).astype("int64")
                label_data = label_data.reshape([-1, 1])
                if batch_id == 0:
                    break
            image.data = core.PaddleBuf(image_data.tolist())
            label.data = core.PaddleBuf(label_data)

            warmup_data.append(image)
            warmup_data.append(label)

            #Enable the int8 quantization
            infer_config.enable_quantizer()
            infer_config.quantizer_config().set_quant_data(warmup_data)
            infer_config.quantizer_config().set_quant_batch_size(100)
            #infer_config.quantizer_config().set_enabled_op_types({"conv2d", "pool2d"});
            predictor = core.create_paddle_predictor(infer_config)
            predictor.SaveOptimModel(self.int8_model_save_path)

            logger.info('Finish InferQuantStrategy::on_compresseion_begin')
