/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        SliceCompute<1>(ctx);
        break;
      case 2:
        SliceCompute<2>(ctx);
        break;
      case 3:
        SliceCompute<3>(ctx);
        break;
      case 4:
        SliceCompute<4>(ctx);
        break;
      case 5:
        SliceCompute<5>(ctx);
        break;
      case 6:
        SliceCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void SliceCompute(const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto in = context.Input<framework::Tensor>("Input");
    auto out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    auto out_dims = out->dims();
    auto in_dims = in->dims();
    auto axes = context.Attr<std::vector<int>>("axes");
    auto starts = context.Attr<std::vector<int>>("starts");

    auto offsets = Eigen::array<int, D>();
    auto extents = Eigen::array<int, D>();
    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = out_dims[i];
    }
    int start;
    for (size_t i = 0; i < axes.size(); ++i) {
      start = starts[i];
      if (start < 0) {
        start = (start + in_dims[axes[i]]);
      }
      start = std::max(start, 0);
      offsets[axes[i]] = start;
    }
    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *out);
    out_t.device(place) = in_t.slice(offsets, extents);
  }
};
}  // namespace operators
}  // namespace paddle
