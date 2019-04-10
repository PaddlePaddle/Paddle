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
        class_dim = 1000
        shape = [1, 3, 224, 224]
        if context.epoch_id == self.end_epoch:
            logger.info('InferQuantStrategy::on_compression_begin')
            test_graph = IrGraph(
                 core.Graph(context.eval_graph.program.desc), for_test=True)
            build_strategy = core.ParallelExecutor.BuildStrategy()
            pass_builder = build_strategy._finalize_strategy_and_create_passes()
            scope = context.scope
            infer_config = core.AnalysisConfig("AnalysisConfig")
            infer_config.switch_ir_optim(True)
            infer_config.disable_gpu
            infer_config.set_model(context.load_model_dir)
            infer_config.enable_mkldnn
            infer_config.pass_builder
            """
            infer_config.pass_builder().set_passes(
               {"infer_clean_graph_pass", "mkldnn_placement_pass",
                "depthwise_conv_mkldnn_pass", "conv_bn_fuse_pass",
                "conv_eltwiseadd_bn_fuse_pass", "conv_bias_mkldnn_fuse_pass",
                "conv_elementwise_add_mkldnn_fuse_pass", "conv_relu_mkldnn_fuse_pass",
                "fc_fuse_pass", "is_test_pass"});   
            """
            warmup_data = []   
            dshape= [3*224*224]
            for batch_id, data in enumerate(context.eval_reader()):
                 image = np.array(map(lambda x: x[0].reshape(dshape), data)).astype(
                     "float32")
                 image = np.reshape(image, (1, np.product(image.shape))) 
                 #label = np.array(map(lambda x: x[1], data)).astype("int64")
                 pre_data=core.PaddleTensor()
                 pre_data.name = "x"
                 pre_data.shape = shape
                 #pre_data.data.resize(50*3*318*318)
                 pre_data.data = core.PaddleBuf(image[0])
                 pre_data.dtype = core.PaddleDType.FLOAT32
                 warmup_data.append(pre_data)
                 if batch_id == 0:
                    break
            infer_config.enable_quantizer();
            infer_config.quantizer_config().set_quant_data(warmup_data);
            infer_config.quantizer_config().set_quant_batch_size(1);
            infer_config.quantizer_config().set_enabled_op_types({"conv2d", "pool2d"});
            core.create_paddle_predictor(infer_config)
     
            """
            context.eval_graph.program = test_graph.to_program()
            logger.info('test_graph to_program')
            if self.int8_model_save_path:
                # convert the weights as int8_t type
                convert_int8_pass = ConvertToInt8Pass(
                    scope=scope, place=context.place)
                convert_int8_pass.apply(test_graph)
                executor = Executor(context.place)
                io.save_inference_model(
                    self.int8_model_save_path,
                    context.eval_graph.in_nodes.keys(), [
                        context.eval_graph.get_var(var_name)
                        for var_name in context.eval_graph.out_nodes.values()
                    ],
                    executor,
                    main_program=test_graph.to_program(),
                    model_filename='model',
                    params_filename='weights',
                    export_for_deployment=False)
            logger.info('Finish InferQuantStrategy::on_compresseion_begin')
            """
