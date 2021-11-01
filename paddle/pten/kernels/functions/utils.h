// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/cpu/utils.h"
#include "paddle/pten/kernels/cuda/utils.h"
namespace pten {

template <typename T>
inline void StridedMemcpyWithAxis0(
    const paddle::platform::DeviceContext& dev_ctx,
    const pten::DenseTensor& input,
    const std::vector<const pten::DenseTensor*>& shape_refer,
    std::vector<pten::DenseTensor*>* outputs) {
  const pten::DDim in_stride = stride_numel(input.dims());
  const int axis = 0;
  size_t input_offset = 0;

  for (size_t i = 0; i < outputs->size(); ++i) {
    auto out_stride = stride_numel(shape_refer[i]->dims());
    auto out = outputs->at(i);
    if (out != nullptr) {
      paddle::operators::StridedNumelCopyWithAxis<T>(
          dev_ctx,
          axis,
          out->mutable_data<T>(),
          out_stride,
          input.data<T>() + input_offset,
          in_stride,
          out_stride[axis]);
    }
    input_offset += out_stride[axis];
  }
}

}  // namespace pten
