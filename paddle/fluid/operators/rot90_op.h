/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
class Rot90Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

template <typename T>
class Rot90Kernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");
    auto k = ctx.template Attr<int>("k");
    auto rot_dims = ctx.template Attr<std::vector<int>>("dims");

    auto total_rot_dims = rot_dims.size();

    auto x_dims = x->dims();
    const int total_dims = x_dims.size();

    TORCH_CHECK(total_rot_dims == 2,
                "expected total rotation dims == 2, but got dims = ",
                total_rot_dims);
    TORCH_CHECK(total_dims >= 2,
                "expected total dims >= 2, but got total dims = ", total_dims);
    TORCH_CHECK(dims[0] != dims[1] && std::abs(dims[0] - dims[1]) != total_dims,
                "expected rotation dims to be different, but got dim0 = ",
                dims[0], " and dim1 = ", dims[1]);
    // check range of dims
    TORCH_CHECK(dims[0] < total_dims && dims[0] >= -total_dims,
                "Rotation dim0 out of range, dim0 = ", dims[0]);
    TORCH_CHECK(dims[1] < total_dims && dims[1] >= -total_dims,
                "Rotation dim1 out of range, dim1 = ", dims[1]);
    // handle modulo with negative k
    k = (4 + (k % 4)) % 4;
    switch (k) {
      case 1:
        return self.flip({dims[1]}).transpose_(dims[0], dims[1]);
      case 2:
        return self.flip(dims);
      case 3:
        return self.flip({dims[0]}).transpose_(dims[0], dims[1]);
      default:
        return self.clone(at::MemoryFormat::Contiguous);
    }

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
