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
#include <iostream>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/clip_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/matrix_bit_code.h"
#include "paddle/fluid/platform/transform.h"
namespace paddle {
namespace operators {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
using platform::Transform;

template <typename DeviceContext, typename T>
class HierarchicalSigmoidOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* w = ctx.Input<framework::Tensor>("W");
    auto* ids = ctx.Input<framework::Tensor>("Ids");
    auto* bias = ctx.Input<framework::Tensor>("Bias");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* pre_out = ctx.Output<framework::Tensor>("PreOut");
    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));
    int64_t code_length = math::FindLastSet(num_classes - 1);
    int64_t batch_size = in->dims()[0];
    framework::Tensor sum;
    math::SetConstant<DeviceContext, T> zero;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto pre_out_data = pre_out->mutable_data<T>(
        framework::make_ddim({batch_size, code_length}), ctx.GetPlace());
    auto pre_out_mat = EigenMatrix<T>::From(*pre_out);
    zero(dev_ctx, pre_out, static_cast<T>(0.0));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    math::RowwiseSum<DeviceContext, T> row_sum;
    math::MatrixBitCodeFunctor<T> bit_code(num_classes, ids->data<int64_t>());

    std::vector<int64_t> sum_dims({batch_size, 1UL});
    sum.mutable_data<T>(framework::make_ddim(sum_dims), ctx.GetPlace());
    auto sum_mat = EigenMatrix<T>::From(sum);
    out->mutable_data<T>(ctx.GetPlace());
    auto out_mat = framework::EigenVector<T>::Flatten(*out);
    if (bias) {
      bit_code.Add(pre_out, *bias);
    }
    bit_code.Mul(pre_out, *w, *in);
    // clip the matrix with (-40, 40)
    Transform<DeviceContext> trans;
    trans(ctx.template device_context<DeviceContext>(), pre_out_data,
          pre_out_data + pre_out->numel(), pre_out_data,
          ClipFunctor<T>(static_cast<T>(-40.0), static_cast<T>(40.0)));
    bit_code.Sum(*pre_out, out, static_cast<T>(-1));
    // softrelu with threshold is 40.0
    trans(ctx.template device_context<DeviceContext>(), pre_out_data,
          pre_out_data + pre_out->numel(), pre_out_data,
          ClipFunctor<T>(static_cast<T>(-40.0), static_cast<T>(40.0)));
    pre_out_mat.device(place) = (static_cast<T>(1.0) + pre_out_mat.exp()).log();
    row_sum(dev_ctx, *pre_out, &sum);
    out_mat.device(place) = sum_mat + out_mat;
  }
};

template <typename DeviceContext, typename T>
class HierarchicalSigmoidGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* w = ctx.Input<framework::Tensor>("W");
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* w_grad = ctx.Output<framework::Tensor>(framework::GradVarName("W"));
    auto* bias_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("Bias"));
    auto* ids = ctx.Input<framework::Tensor>("Ids");
    auto* pre_out = ctx.Input<framework::Tensor>("PreOut");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));
    int64_t code_length = math::FindLastSet(num_classes - 1);
    int64_t batch_size = in->dims()[0];
    framework::Tensor pre_out_grad;
    pre_out_grad.mutable_data<T>(
        framework::make_ddim({batch_size, code_length}), ctx.GetPlace());
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto pre_out_mat = EigenMatrix<T>::From(*pre_out);
    auto pre_out_grad_mat = EigenMatrix<T>::From(pre_out_grad);
    math::MatrixBitCodeFunctor<T> bit_code(num_classes, ids->data<int64_t>());
    // softrelu derivative
    bit_code.OutGrad(&pre_out_grad, *out_grad);
    pre_out_grad_mat.device(place) =
        pre_out_grad_mat *
        (static_cast<T>(1.0) - static_cast<T>(1.0) / pre_out_mat.exp());
    bit_code.Sub(&pre_out_grad);
    if (bias_grad) {
      bias_grad->mutable_data<T>(ctx.GetPlace());
      bit_code.AddGrad(pre_out_grad, bias_grad);
    }
    in_grad->mutable_data<T>(ctx.GetPlace());
    w_grad->mutable_data<T>(ctx.GetPlace());
    bit_code.MulGradWeight(pre_out_grad, w_grad, *in);
    bit_code.MulGradError(pre_out_grad, *w, in_grad);
  }
};

}  // namespace operators
}  // namespace paddle
