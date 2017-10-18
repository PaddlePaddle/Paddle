/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class BatchNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<Tensor>("X");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* bias = ctx.Input<Tensor>("Bias");
    const bool is_test = ctx.Attr<bool>("is_test");

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto& x_dims = x->dims();
    PADDLE_ENFORCE(x_dims.size() >= 3 && x_dims.size() <= 5,
                   "The Input dim size should be between 3 and 5");
    const int N = x_dims[0];                          // sample num / batch size
    const int C = x_dims[1];                          // channel num
    const int H = x_dims[2];                          // sample height
    const int W = x_dims.size() > 3 ? x_dims[3] : 1;  // sample width
    // sample depth, the last dimension or 1
    const int D = x_dims.size() > 4 ? x_dims[4] : 1;

    const int sample_size = H * W * D;

    const auto& place = ctx.GetEigenDevice<Place>();
    auto* out = ctx.Output<Tensor>("Out");
    auto* mean_out = ctx.Output<Tensor>("MeanOut");
    auto* variance_out = ctx.Output<Tensor>("VarianceOut");
    auto* saved_mean = ctx.Output<Tensor>("SavedMean");
    auto* saved_variance = ctx.Output<Tensor>("SavedVariance");

    // alloc memory
    out->mutable_data<T>(ctx.GetPlace());
    mean_out->mutable_data<T>(ctx.GetPlace());
    variance_out->mutable_data<T>(ctx.GetPlace());
    saved_mean->mutable_data<T>(ctx.GetPlace());
    saved_variance->mutable_data<T>(ctx.GetPlace());
  }
};

template <typename Place, typename T>
class BatchNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};
}  // namespace operators
}  // namespace paddle
