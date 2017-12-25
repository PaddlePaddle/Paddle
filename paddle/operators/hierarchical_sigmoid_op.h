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
#include "paddle/operators/clip_op.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/matrix_bit_code.h"
#include "paddle/platform/transform.h"

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
    auto* params = ctx.Input<framework::Tensor>("Parameters");
    auto* label = ctx.Input<framework::Tensor>("Label");
    auto* bias = ctx.Input<framework::Tensor>("Bias");
    auto* out = ctx.Output<framework::Tensor>("Out");
    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));

    int64_t code_length = math::FindLastSet(num_classes - 1);
    int64_t batch_size = in->dims()[0];
    auto* ids = label->data<int64_t>();
    framework::Tensor pre_out;
    framework::Tensor sum;
    auto pre_out_data = pre_out.mutable_data<T>(
        framework::make_ddim({batch_size, code_length}), ctx.GetPlace());
    auto pre_out_mat = EigenMatrix<T>::From(pre_out);

    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto& device_ctx = ctx.template device_context<DeviceContext>();
    math::RowwiseSum<DeviceContext, T> row_sum;
    math::MatrixBitCodeFunctor<T> bit_code;

    std::vector<int64_t> sum_dims({batch_size, 1UL});
    sum.mutable_data<T>(framework::make_ddim(sum_dims), ctx.GetPlace());
    auto sum_mat = EigenMatrix<T>::From(sum);
    out->mutable_data<T>(ctx.GetPlace());
    auto out_mat = framework::EigenVector<T>::Flatten(*out);

    if (bias) {
      bit_code.Add(num_classes, ids, pre_out, *bias);
    }
    for (int i = 0; i < in->dims()[0]; ++i) {
      bit_code.Mul(num_classes, ids, pre_out, params->Slice(i, i + 1),
                   in->Slice(i, i + 1));
    }
    // clip the matrix with (-40, 40)
    Transform<DeviceContext> trans;
    trans(ctx.template device_context<DeviceContext>(), pre_out_data,
          pre_out_data + pre_out.numel(), pre_out_data,
          ClipFunctor<T>(static_cast<T>(-40.0), static_cast<T>(40.0)));
    bit_code.Sum(num_classes, ids, pre_out, *out, static_cast<T>(-1));
    // softrelu with threshold is 40.0
    trans(ctx.template device_context<DeviceContext>(), pre_out_data,
          pre_out_data + pre_out.numel(), pre_out_data,
          ClipFunctor<T>(static_cast<T>(-40.0), static_cast<T>(40.0)));
    pre_out_mat.device(place) = (static_cast<T>(1.0) + pre_out_mat.exp()).log();

    row_sum(device_ctx, pre_out, &sum);
    out_mat.device(place) = sum_mat + out_mat;
  }
};

template <typename DeviceContext, typename T>
class HierarchicalSigmoidGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* params =
        ctx.Output<framework::Tensor>(framework::GradVarName("Parameters"));
    auto* bias = ctx.Output<framework::Tensor>(framework::GradVarName("Bias"));
    auto* label = ctx.Input<framework::Tensor>("Label");
    size_t num_classes = static_cast<size_t>(ctx.Attr<int>("num_classes"));
    int64_t code_length = math::FindLastSet(num_classes - 1);
    int64_t batch_size = in->dims()[0];

    framework::Tensor pre_out;
    pre_out.mutable_data<T>(framework::make_ddim({batch_size, code_length}),
                            ctx.GetPlace());
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto& device_ctx = ctx.template device_context<DeviceContext>();
    auto pre_out_mat = EigenMatrix<T>::From(pre_out);
    auto* ids = label->data<int64_t>();

    // init pre_out matrix with {1.0}
    math::SetConstant<DeviceContext, T> one;
    math::MatrixBitCodeFunctor<T> bit_code;
    one(device_ctx, &pre_out, static_cast<T>(1.0));
    // softrelu derivative
    pre_out_mat.device(place) =
        pre_out_mat * (static_cast<T>(1.0) - static_cast<T>(1.0) / pre_out_mat);

    bit_code.Sub(num_classes, ids, pre_out);

    if (bias) {
      bit_code.AddGrad(num_classes, ids, pre_out, *bias);
    }

    for (int i = 0; i < in_grad->dims()[0]; ++i) {
      auto p_sliced = params->Slice(i, i + 1);
      auto in_sliced = in->Slice(i, i + 1);
      auto in_grad_sliced = in_grad->Slice(i, i + 1);
      bit_code.MulGradWeight(num_classes, ids, pre_out, p_sliced, in_sliced);
      bit_code.MulGradError(num_classes, ids, pre_out, p_sliced,
                            in_grad_sliced);
    }
  }
};

}  // namespace operators
}  // namespace paddle
