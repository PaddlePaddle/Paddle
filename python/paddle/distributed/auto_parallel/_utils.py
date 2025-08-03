# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


def _in_auto_parallel_align_mode_handle_none_gradients_in_step(step_method):
    def wrapper(self, *args, **kwargs):
        if not isinstance(self._parameter_list[0], dict):
            for param in self._parameter_list:
                if param.stop_gradient or param.grad is not None:
                    continue

                if hasattr(param, "main_grad"):
                    param.main_grad = paddle.zeros_like(
                        param, dtype=paddle.float32
                    )
                else:
                    param.grad = paddle.zeros_like(param, dtype=param.dtype)
        else:
            for param_group in self._param_groups:
                for param in param_group['params']:
                    if param.stop_gradient or param.grad is not None:
                        continue

                    if hasattr(param, "main_grad"):
                        param.main_grad = paddle.zeros_like(
                            param, dtype=paddle.float32
                        )
                    else:
                        param.grad = paddle.zeros_like(param, dtype=param.dtype)
        return step_method(self, *args, **kwargs)

    return wrapper
