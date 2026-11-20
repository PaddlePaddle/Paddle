// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// paddle_tensor.h —— 收发 paddle::Tensor 的两个接口（实现在 paddle_tensor.cc，
// Python 绑定在 paddle/fluid/pybind/gin_overlap_api.cc）：
//   - send_window_tensor：把本组 send window 的一段包成 1-D fp32
//     paddle::Tensor。零拷贝，Paddle 不负责释放；各桶的 grad_storage 就是它。
//   - copy_2d：在 Paddle 当前计算流上做一次 pitched 拷贝。
// 两者出错时抛异常，不返回错误码。
#pragma once

#include <cstddef>

#include "paddle/phi/api/include/tensor.h"

namespace gin_overlap {

paddle::Tensor send_window_tensor(int group,
                                  size_t send_offset,
                                  long long numel);  // NOLINT

void copy_2d(const paddle::Tensor& dst,
             size_t dst_offset,
             size_t dst_pitch,
             const paddle::Tensor& src,
             size_t src_offset,
             size_t src_pitch,
             size_t width,
             size_t height);

}  // namespace gin_overlap
