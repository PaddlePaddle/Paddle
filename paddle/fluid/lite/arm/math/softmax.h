// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename T>
void softmax_basic(const T* din, T* dout, const int axis_size,
                   const int inner_num, const int outer_num);

template <typename T>
void softmax_inner8_axis4(const T* din, T* dout, const int axis_size,
                          const int inner_num, const int outer_num);

template <typename T>
void softmax_inner4_axis4(const T* din, T* dout, const int axis_size,
                          const int inner_num, const int outer_num);
template <typename T>
void softmax_inner8(const T* din, T* dout, const int axis_size,
                    const int inner_num, const int outer_num);

template <typename T>
void softmax_inner4(const T* din, T* dout, const int axis_size,
                    const int inner_num, const int outer_num);

template <typename T>
void softmax_inner1_large_axis(const T* din, T* dout, const int outer_size,
                               const int axis_size);

template <typename T>
void softmax_inner1_small_axis(const T* din, T* dout, const int outer_size,
                               const int axis_size);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
