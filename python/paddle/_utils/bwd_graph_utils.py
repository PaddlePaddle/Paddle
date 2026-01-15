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
from paddle.base.wrapped_decorator import signature_safe_contextmanager

from .download import check_and_create_dir


@signature_safe_contextmanager
def capture_backward_subgraph_guard(
    dump_dir_path: str, need_dump_grad_tensors: bool = False
):
    assert dump_dir_path is not None, "The dump_dir_path should not be None"
    check_and_create_dir(dump_dir_path)
    paddle.base.core.eager._start_capture_backward_viz_subgraph(
        dump_dir_path, need_dump_grad_tensors
    )
    try:
        yield
    finally:
        paddle.base.core.eager._stop_capture_backward_viz_subgraph()
