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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/matrix_bit_code.h"

namespace paddle {
namespace operators {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class HierarchicalSigmoidOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto params = ctx.MultiInput<framework::Tensor>("Parameters");
    auto* label = ctx.Input<framework::Tensor>("Label");
    auto* bias = ctx.Input<framework::Tensor>("Bias");
    auto* out = ctx.Output<framework::Tensor>("Out");
    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));

    framework::Tensor sum;
    framework::Tensor pre_out;
    auto place = ctx.GetEigenDevice<Place>();
    auto& device_ctx = ctx.device_context();
    math::ColwiseSum<Place, T> col_sum;
    math::RowwiseSum<Place, T> row_sum;

    auto pre_out_mat = EigenMatrix<T>::From(pre_out);
    int64_t batch_size = ins[0]->dims()[0];
    int64_t size = ins.size();

    std::vector<int64_t> pre_out_dims({batch_size, size});
    pre_out.mutable_data<T>(framework::make_ddim(pre_out_dims), ctx.GetPlace());
    std::vector<int64_t> sum_dims({batch_size, 1UL});
    sum.mutable_data<T>(framework::make_ddim(sum_dims), ctx.GetPlace());
    out->mutable_data<T>(ctx.GetPlace());

    if (bias) {
      math::AddByBitCode<T>(num_classes, *label, pre_out, *bias);
    }

    for (size_t i = 0; i < ins.size(); ++i) {
      math::MulByBitCode<T>(num_classes, *label, pre_out, *params[i], *ins[i]);
    }
    // clip the matrix with (-40, 40)
    pre_out_mat.device(place) =
        pre_out_mat.abs().cwiseMax(static_cast<T>(40.0));
    math::SumByBitCode<T>(num_classes, *label, *out, pre_out,
                          static_cast<T>(-1));
    // softrelu
    pre_out_mat.device(place) = (static_cast<T>(1) + pre_out_mat.exp()).log();

    row_sum(device_ctx, pre_out, &sum);
    col_sum(device_ctx, *out, &sum);
  }
};

template <typename Place, typename T>
class HierarchicalSigmoidGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
