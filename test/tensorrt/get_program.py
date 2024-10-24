# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

import paddle
from paddle import nn, static
from paddle.nn import TransformerEncoder, TransformerEncoderLayer


def get_r50_program():
    paddle.enable_static()
    from paddle.vision.models import wide_resnet50_2

    with paddle.pir_utils.IrGuard():
        infer_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with static.program_guard(infer_program, startup_program):
            scope = paddle.static.global_scope()
            input_data = paddle.static.data(
                shape=[-1, 3, 224, 224], dtype='float32', name='input'
            )
            model = wide_resnet50_2()
            model.eval()
            output = model(input_data)
        place = paddle.CUDAPlace(0)
        exe = static.Executor(place)
        exe.run(startup_program)

    params = infer_program.global_block().all_parameters()
    param_dict = {}
    for v in params:
        name = v.get_defining_op().attrs()["parameter_name"]
        param_dict.update({name: np.array(scope.var(name).get_tensor())})

    return infer_program, scope, param_dict


def get_dummy_program():
    paddle.enable_static()
    with paddle.pir_utils.IrGuard():
        main_program = paddle.static.Program()
        default_startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, default_startup_program):
            scope = paddle.static.global_scope()
            input = paddle.static.data(
                shape=[-1, 64], dtype='float32', name='input'
            )
            weight_numpy = np.random.rand(64, 64).astype('float32')
            weight = paddle.create_parameter(
                name="w",
                shape=[64, 64],
                dtype='float32',
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(weight_numpy)
                ),
            )
            bias_numpy = np.random.rand(64).astype('float32')
            bias = paddle.create_parameter(
                name="b",
                shape=[64],
                dtype='float32',
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(bias_numpy)
                ),
            )
            x = paddle.matmul(input, weight)
            x_1 = paddle.add(x, bias)
            x_1 = paddle.unsqueeze(x_1, axis=0)
            x_1 = paddle.squeeze(x_1, axis=0)
            y = paddle.nn.functional.relu(x_1)
            y_gelu_1 = paddle.nn.functional.gelu(y)
            y_gelu_2 = paddle.nn.functional.gelu(x_1)

            # Concatenate the outputs of the two GELU operations
            concat_out = paddle.concat([y_gelu_1, y_gelu_2], axis=-1)
            output = paddle.unsqueeze(concat_out, axis=0)

        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(default_startup_program)

        params = main_program.global_block().all_parameters()
        param_dict = {}
        # save parameters
        for v in params:
            name = v.get_defining_op().attrs()["parameter_name"]
            param_dict.update({name: np.array(scope.var(name).get_tensor())})
    return main_program, scope, param_dict


class BertModel(nn.Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = TransformerEncoderLayer(
            hidden_size, num_attention_heads, hidden_size * 4
        )
        self.encoder = TransformerEncoder(encoder_layer, num_hidden_layers)

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        encoded_output = self.encoder(embeddings)
        return encoded_output


def get_bert_program():
    paddle.enable_static()

    vocab_size = 30522  # BERT base vocab size
    hidden_size = 768
    num_hidden_layers = 2
    num_attention_heads = 12
    seq_length = 128

    with paddle.pir_utils.IrGuard():
        main_program = static.default_main_program()
        startup_program = static.default_startup_program()
        with static.program_guard(main_program, startup_program):
            scope = paddle.static.global_scope()
            input_ids = static.data(
                name='input_ids', shape=[-1, -1], dtype='int64'
            )

            bert_model = BertModel(
                vocab_size, hidden_size, num_hidden_layers, num_attention_heads
            )
            bert_model.eval()
            logits = bert_model(input_ids)

    place = (
        paddle.CUDAPlace(0)
        if paddle.is_compiled_with_cuda()
        else paddle.CPUPlace()
    )
    pir_program = main_program

    with paddle.pir_utils.IrGuard():
        with paddle.static.program_guard(pir_program, startup_program):
            x = np.ones([1, seq_length]).astype('int64')
            executor = paddle.static.Executor(place)
            executor.run(startup_program)
            fetches = executor.run(
                pir_program,
                feed={"input_ids": x},
                fetch_list=pir_program.list_vars()[-3],
            )

    params = main_program.global_block().all_parameters()
    param_dict = {}
    # save parameters
    for v in params:
        name = v.get_defining_op().attrs()["parameter_name"]
        param_dict.update({name: np.array(scope.var(name).get_tensor())})
    return pir_program, scope, param_dict


class SimpleGatherNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(149600, 1)

    def forward(self, map_vector_features, polyline_mask):
        map_vector_features = map_vector_features[polyline_mask]

        return map_vector_features
