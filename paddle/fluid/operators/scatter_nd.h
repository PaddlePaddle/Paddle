/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <cstring>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"
#include "unordered_set"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

/**
 * index can be set to N dimension
 * if dim=0: out[index[i][j]][j] = update[i][j]
 * if dim=1: out[i][index[i][j]] = update[i][j]
 * ...
 */
template <typename T, typename IndexT>
void ScatterNDAssign(const platform::DeviceContext& ctx, const Tensor& src,
                     const Tensor& index, Tensor* output, int dim) {
  PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()));

  int index_size = framework::product(index.dims());
  auto index_dims = index.dims();
  auto src_dims = src.dims();
  auto dst_dims = output->dims();

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // check src shape and dst shape should match (expect `dim` dimension)
  for (int i = 0; i < src_dims.size(); i++) {
    if (i != dim) {
      PADDLE_ENFORCE(src_dims[i] == dst_dims[i]);
    }
  }
  int slice_size = 1;
  for (int i = dim + 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];
  int update_size = index_size / slice_size;
  int output_batch = framework::product(dst_dims) / slice_size;

  const size_t slice_bytes = slice_size * sizeof(T);
  const size_t dst_slice_bytes = output_batch * sizeof(T);
  const size_t src_slice_bytes = slice_size * index_dims[dim] * sizeof(T);

  int j = 0;
  for (int i = 0; i < update_size; ++i) {
    int i_max = j + index_dims[dim];
    for (int j = 0; j < i_max && j < index_size; ++j) {
      IndexT index_ = p_index[j];
      memcpy(p_output + i * dst_slice_bytes +
                 (j - i_max + index_dims[dim]) * slice_bytes,
             p_src + i * src_slice_bytes + index_ * slice_bytes, slice_bytes);
    }
  }
}

}  // namespace operators
}  // namespace paddle
