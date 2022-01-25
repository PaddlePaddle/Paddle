// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TraceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* out = context.Output<framework::Tensor>("Out");

    const int64_t offset = context.Attr<int>("offset");
    const int64_t dim1 = context.Attr<int>("axis1");
    const int64_t dim2 = context.Attr<int>("axis2");

    auto output_dims = out->dims();

    T* out_data = out->mutable_data<T>(context.GetPlace());

    const framework::Tensor diag =
        Diagonal<DeviceContext, T>(context, input, offset, dim1, dim2);
    if (diag.numel() > 0) {
      auto x = framework::EigenMatrix<T>::Reshape(diag, diag.dims().size() - 1);
      auto output = framework::EigenVector<T>::Flatten(*out);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({1});
      output.device(place) = x.sum(reduce_dim);
      out->Resize(output_dims);
    } else {
      std::fill(out_data, out_data + out->numel(), static_cast<T>(0));
    }
  }
};

template <typename DeviceContext, typename T>
class TraceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_x =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    int64_t offset = context.Attr<int>("offset");
    int64_t dim1 = context.Attr<int>("axis1");
    int64_t dim2 = context.Attr<int>("axis2");
  }
};

}  // namespace operators
}  // namespace paddle
