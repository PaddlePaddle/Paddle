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


def miminize_bfgs(func,
                  jac,
                  x0,
                  dtype='float32',
                  H0=None,
                  epsilon=1e-8,
                  max_iters=50,
                  max_line_search_iters=50,
                  summary_only=True,
                  name=None):
    k = 0

    while norm(gradient_fk) > epsilon:
        pk = -H_prev * gradient_fk
        alpha_k = line_search()
        x_new = x_prev + alpha_k * pk
        sk = x_new - x_prev
        yk = gradient(x_new) - gradient(x_prev)
        rhok = 1
        H_new = 1
        k = k + 1
