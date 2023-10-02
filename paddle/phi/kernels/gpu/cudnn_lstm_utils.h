// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/gpu/cudnn_lstm_cache.h"
#endif
#ifdef PADDLE_WITH_HIP
#include "paddle/phi/kernels/gpu/miopen_lstm_cache.h"
#endif

namespace phi {

template <typename T, typename Type>
inline bool is_continuous(const Type &weight_list) {
  bool continuous = true;
  for (size_t i = 0; i < weight_list.size() - 1; ++i) {
    auto *in_data = weight_list[i]->template data<T>();
    auto *in_after_data = weight_list[i + 1]->template data<T>();
    auto in_size = weight_list[i]->numel();
    bool temp = in_data + in_size == in_after_data;
    continuous = continuous && temp;
  }
  return continuous;
}

inline int size_sum(const std::vector<const phi::DenseTensor *> &weight_list) {
  int size = 0;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    auto in_size = weight_list[i]->numel();
    size += in_size;
  }
  return size;
}

template <typename T>
inline void weight_to_tensor(
    const phi::Place &place,
    gpuStream_t stream,
    const std::vector<const phi::DenseTensor *> &weight_list,
    phi::DenseTensor *weight) {
  auto weight_data = weight->data<T>();
  int weight_offset = 0;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    const T *in_data = weight_list[i]->data<T>();
    auto in_size = weight_list[i]->numel();

    memory_utils::Copy(weight->place(),
                       weight_data + weight_offset,
                       weight_list[i]->place(),
                       in_data,
                       in_size * sizeof(T),
                       stream);
    weight_offset += in_size;
  }
}

}  // namespace phi
