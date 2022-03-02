# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from .line_search import strong_wolfe


def miminize_bfgs(f,
                  x0,
                  max_iter=50,
                  tolerance_grad=1e-8,
                  tolerance_change=0,
                  H0=None,
                  norm_type=np.inf,
                  line_search_method='strong_wolfe',
                  max_line_search_iters=50,
                  initial_step_length=1.0,
                  dtype='float32',
                  name=None):
    k = 0
    I = paddle.eye(input_dim, dtype=dtype)
    while paddle.norm(gradient_fk, norm_type) > epsilon and k < max_iters:
        pk = -H_prev * gradient_fk
        alpha_k, gradient(x_new) = line_search(func, gradient)
        x_new = x_prev + alpha_k * pk
        sk = x_new - x_prev
        yk = gradient(x_new) - gradient(x_prev)
        if norm(gradient_new) <= epsilon:
            break

        rhok = 1. / paddle.dot(yk, sk)
        Vk_transpose = I - rhok * sk * yk
        Vk = I - rhok * yk * sk
        H_k = paddle.dot(paddle.dot(Vk_transpose, H_preve), Vk) + rhok * sk * sk
        k += 1
