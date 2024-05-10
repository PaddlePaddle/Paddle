# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import paddle

paddle.enable_static()


main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()
with paddle.static.program_guard(main_program, startup_program):
    type = "float8_e4m3fn"

    input1 = paddle.ones([16, 16], dtype='float32')
    input2 = paddle.ones([16, 16], dtype='float32')

    # input1 = paddle.static.data(name='x', shape=[16,16],dtype="float32")
    # input2 = paddle.static.data(name='y', shape=[16,16],dtype="float32")
    # output0 = paddle.matmul(paddle.cast(input1,dtype), paddle.cast(input2,dtype))

    output0 = paddle.linalg.fp8_fp8_fp16_gemm_fused(paddle.cast(input1,type), paddle.cast(input2,type))
    print("output0: ",output0)


    b = paddle.create_parameter(
        shape=[16, 16],
        dtype='float16',
        default_initializer=paddle.nn.initializer.Constant(1.0),
    )

    output1 = paddle.add(output0, b)

    exe = paddle.static.Executor(paddle.CUDAPlace(0))

    exe.run(startup_program)

    paddle.static.save_inference_model(
        "./model/model", [input1, input2], [output1], exe
    )

import os

import paddle

place = paddle.CPUPlace()
scope = paddle.static.global_scope()
exe = paddle.static.Executor(place)

dirname = os.path.dirname("./model/model")
basename = os.path.basename("./model")
model_filename = basename + ".pdmodel"
params_filename = basename + ".pdiparams"

[
    infer_program,
    feed_target_names,
    fetch_targets,
] = paddle.static.load_inference_model(
    path_prefix=dirname,
    executor=exe,
    model_filename=model_filename,
    params_filename=params_filename,
)


'''
params_kv = infer_program.state_dict()
for name in params_kv:
    param = params_kv[name]
    print(name)
    print(np.array(param))
'''
for name in feed_target_names:
    print("feed_target_names: ", name)

for name in infer_program.list_vars():
    print("var_name: ", name)
