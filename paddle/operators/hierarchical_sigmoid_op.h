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
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* param = ctx.Input<framework::Tensor>("Parameter");
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
    int64_t code_length = math::FindLastSet(num_classes - 1);

    std::vector<int64_t> pre_out_dims({batch_size, code_length});
    pre_out.mutable_data<T>(framework::make_ddim(pre_out_dims), ctx.GetPlace());
    std::vector<int64_t> sum_dims({batch_size, 1UL});
    sum.mutable_data<T>(framework::make_ddim(sum_dims), ctx.GetPlace());
    out->mutable_data<T>(ctx.GetPlace());

    if (bias) {
      math::AddByBitCode<T>(num_classes, *label, pre_out, *bias);
    }

    for (size_t i = 0; i < in.dims()[0]; ++i) {
      math::MulByBitCode<T>(num_classes, *label, pre_out,
                            *params->Slice(i, i + 1), *in->Slice(i, i + 1));
    }
    // clip the matrix with (-40, 40)
    pre_out_mat.device(place) =
        pre_out_mat.abs().cwiseMax(static_cast<T>(40.0));
    math::SumByBitCode<T>(num_classes, *label, *out, pre_out,
                          static_cast<T>(-1));

    // softrelu with threshold is 40.0
    pre_out_mat.device(place) =
        pre_out_mat.abs().cwiseMax(static_cast<T>(40.0));
    pre_out_mat.device(place) = (static_cast<T>(1.0) + pre_out_mat.exp()).log();

    row_sum(device_ctx, pre_out, &sum);
    col_sum(device_ctx, *out, &sum);
  }
};

template <typename Place, typename T>
class HierarchicalSigmoidGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* params =
        ctx.Output<framework::Tensor>(framework::GradVarName("Parameters"));
    auto* bias = ctx.Output<framework::Tensor>(framework::GradVarName("Bias"));
    auto* label =
        ctx.Output<framework::Tensor>(framework::GradVarName("Label"));
    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));

    framework::Tensor pre_out;
    auto place = ctx.GetEigenDevice<Place>();
    auto& dev_ctx = ctx.device_context();
    int64_t batch_size = in_grad.dims()[0];
    int64_t code_length = math::FindLastSet(num_classes - 1);
    auto pre_out_mat = EigenMatrix<T>::From(pre_out);

    // init pre_out matrix with {1.0}
    std::vector<int64_t> pre_out_dims({batch_size, code_length});
    pre_out.mutable_data<T>(framework::make_ddim(pre_out_dims), ctx.GetPlace());
    math::SetConstant<Place, T> set;
    set(dev_ctx, &pre_out, static_cast<T>(1.0));
    // softrelu derivative
    pre_out_mat.device(place) =
        pre_out_mat * (static_cast<T>(1.0) - static_cast<T>(1.0) / pre_out_mat);

    math::SubByBitCode<T>(num_classes, *label, pre_out);

    if (bias) {
      math::AddByBitCodeGrad<T>(num_classes, *label, pre_out, *bias);
    }

    for (size_t i = 0; i < in_grad.dims()[0]; ++i) {
      math::MulByBitCodeGradWeight<T>(num_classes, *label, pre_out, *params[i],
                                      *in[i]->Slice(i, i + 1));
      math::MulByBitCodeGradError<T>(num_classes, *label, pre_out, *params[i],
                                     *ins_grad[i]->Slice(i, i + 1));
    }
  }
};

}  // namespace operators
}  // namespace paddle
