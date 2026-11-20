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

// paddle_tensor.h —— 碰 paddle::Tensor 的那两个入口（实现在
// paddle_tensor.cc）：
//   send_window_tensor  本组 send window 的一段包成 1-D fp32 paddle::Tensor
//     （零拷贝，Paddle 不接管释放；各桶的 grad_storage 就是它）。
//   copy_2d  Paddle 当前计算流上一次 pitched 拷贝。
// 这两个按 Python 的习惯直接抛异常（不是像引擎的其它方法那样返回返回码），
// 导出给 Python 的 def 写在 paddle/fluid/pybind/gin_overlap_api.cc 里。
//
// PaddleFleet 的扩展里这个头文件故意不出现 paddle::Tensor（那边的 pybind 入口由
// 外部 host 编译器编，碰不了带 thrust 的 Paddle 头文件），所以只声明了一个
// 「去 .cu 里注册」的函数。搬进 Paddle 之后没有这个限制，直接声明真签名。
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
