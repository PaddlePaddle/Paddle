/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <bitset>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

constexpr size_t dim_bitset_size = 64;

template <typename DeviceContext, typename T>
class FlipKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

template <typename T>
class FlipKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");
    auto flip_dims = ctx.template Attr<std::vector<int>>("dims");

    auto x_dims = x->dims();
    const int total_dims = x_dims.size();
    std::bitset<dim_bitset_size> dim_bitset;
    for (size_t i = 0; i < flip_dims.size(); ++i) {
      int dim = flip_dims[i];
      if (flip_dims[i] < 0) {
        dim += total_dims;
      }
      dim_bitset[dim] = true;
    }
    auto x_strides = framework::stride(x_dims);
    auto numel = x->numel();
    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < numel; ++i) {
      int64_t cur_indices = i;
      int64_t rem = 0;
      int64_t dst_offset = 0;

      for (int d = 0; d < total_dims; ++d) {
        int64_t temp = cur_indices;
        cur_indices = cur_indices / x_strides[d];
        rem = temp - cur_indices * x_strides[d];
        dst_offset += dim_bitset[d]
                          ? (x_dims[d] - 1 - cur_indices) * x_strides[d]
                          : cur_indices * x_strides[d];
        cur_indices = rem;
      }
      out_data[i] = x_data[dst_offset];
    }
  }
};

}  // namespace operators
}  // namespace paddle
