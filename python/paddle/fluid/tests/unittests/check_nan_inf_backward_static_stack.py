# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


# 定义静态图模型
def static_model(x, y):
    z = paddle.pow(x, y)
    return z


def main():
    # 开启静态图模式
    paddle.enable_static()
    paddle.set_flags({"FLAGS_check_nan_inf": 1, "FLAGS_check_nan_inf_level": 0})
    # 定义输入变量和模型输出变量
    x_static = paddle.static.data(name='x_static', shape=[3], dtype='float32')
    y_static = paddle.static.data(name='y_static', shape=[3], dtype='float32')
    x_static.stop_gradient = False
    z_static = static_model(x_static, y_static)

    # 计算梯度
    grads_static = paddle.static.gradients(z_static, x_static, y_static)

    # 创建 Executor 对象
    exe_static = paddle.static.Executor(paddle.CPUPlace())

    # 编译计算图
    exe_static.run(paddle.static.default_startup_program())

    # 执行前向计算和反向传播
    grads_val_static = exe_static.run(
        paddle.static.default_main_program(),
        feed={'x_static': [1, 0, 3], 'y_static': [0, 0, 0]},
        fetch_list=[grads_static],
    )


if __name__ == "__main__":
    main()
