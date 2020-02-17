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
#include <memory.h>
#include <cstring>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using framework::Tensor;

/**
 * A thin wrapper for gathering on cpu tensor
 * Return a new tensor from source tensor, gathered according to indice
 * input[src]: type-T source Tensor
 * input[indice]: type-IndexT indice Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void CPUGather(const platform::DeviceContext& ctx, const Tensor& src,
               const Tensor& indice, Tensor* output) {
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true);
  // check indice of shape 1-D
  if (indice.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(indice.dims()[1], 1,
                      "indice.dims()[1] shold be 1 when indice.dims().size() == "
                      "2 in gather_op.");
  } else {
    PADDLE_ENFORCE_EQ(indice.dims().size(), 1,
                      "indice.dims().size() shold be 1 or 2 in gather_op.");
  }
  int64_t indice_size = indice.dims()[0];

  auto src_dims = src.dims();

  const T* p_src = src.data<T>();
  const IndexT* p_indice = indice.data<IndexT>();
  T* p_output = output->data<T>();

  // slice size
  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t slice_bytes = slice_size * sizeof(T);

  for (int64_t i = 0; i < indice_size; ++i) {
    IndexT indice_ = p_indice[i];
    memcpy(p_output + i * slice_size, p_src + indice_ * slice_size, slice_bytes);
  }
}

template <typename T, typename IndexT = int>
void CPUGatherNd(const platform::DeviceContext& ctx, const Tensor& input,
                 const Tensor& indice, Tensor* output) {
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                    "It shold be running on the CPU");

  auto indice_dims = indice.dims();
  auto indice_dims_size = indice_dims.size();
  auto input_dims = input.dims();
  auto input_dims_size = input_dims.size();

  const T* p_input = input.data<T>();
  const IndexT* p_indice = indice.data<IndexT>();
  T* p_output = output->data<T>();

  // final dim
  int64_t end_size = indice_dims[indice_dims_size - 1];
  // remain dim
  auto remain_ddim = framework::slice_ddim(indice_dims, 0, indice_dims_size - 1);
  int64_t remain_numel = framework::product(remain_ddim);
  // slice size
  int64_t slice_size = 1;
  for (int64_t i = end_size; i < input_dims_size; ++i) {
    slice_size *= input_dims[i];
  }
  const size_t slice_bytes = slice_size * sizeof(T);

  for (int64_t i = 0; i < remain_numel; ++i) {
    int64_t indice_ = 0;
    int64_t temp = 1;
    for (int64_t j = end_size - 1; j >= 0; --j) {
      IndexT indice_value = p_indice[i * end_size + j];
      PADDLE_ENFORCE_LT(indice_value, input_dims[j],
                        "Input(indice[-1)] has wrong value, it is %d",
                        indice_value);
      PADDLE_ENFORCE_GE(indice_value, 0UL,
                        "The value of Input(indice) must be no less than 0");

      indice_ += (indice_value * temp);
      temp *= input_dims[j];
    }
    memcpy(p_output + i * slice_size, p_input + indice_ * slice_size,
           slice_bytes);
  }
}

}  // namespace operators
}  // namespace paddle
