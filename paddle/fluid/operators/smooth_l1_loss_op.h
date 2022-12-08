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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct SmoothL1LossForward {
  HOSTDEVICE SmoothL1LossForward(const T& sigma2) : sigma2(sigma2) {}

  HOSTDEVICE T operator()(const T& val) const {
    T abs_val = std::abs(val);
    if (abs_val < 1.0 / sigma2) {
      return 0.5 * val * val * sigma2;
    } else {
      return abs_val - 0.5 / sigma2;
    }
  }

  T sigma2;
};

template <typename DeviceContext, typename T, typename AttrType = T>
class SmoothL1LossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<phi::DenseTensor>("X");
    auto* in1 = context.Input<phi::DenseTensor>("Y");
    auto* in2 = context.Input<phi::DenseTensor>("InsideWeight");
    auto* in3 = context.Input<phi::DenseTensor>("OutsideWeight");
    auto* out0 = context.Output<phi::DenseTensor>("Diff");
    auto* out1 = context.Output<phi::DenseTensor>("Out");

    out0->mutable_data<T>(context.GetPlace());
    out1->mutable_data<T>(context.GetPlace());
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();

    auto sigma = static_cast<T>(context.Attr<AttrType>("sigma"));
    T sigma2 = sigma * sigma;
    bool has_weight = (in2 != nullptr) && (in3 != nullptr);

    auto x = EigenVector<T>::Flatten(*in0);
    auto y = EigenVector<T>::Flatten(*in1);
    auto diff = EigenVector<T>::Flatten(*out0);

    diff.device(*place) = x - y;
    // multiply inside weight
    if (has_weight) {
      auto inside_weight = EigenVector<T>::Flatten(*in2);
      // cache diff, reused in bp
      diff.device(*place) = diff * inside_weight;
    }

    auto in_counts = in0->numel();
    Tensor ptensor_errors;
    ptensor_errors.mutable_data<T>({static_cast<int>(in_counts)},
                                   context.GetPlace());
    auto errors = EigenVector<T>::Flatten(ptensor_errors);
    // apply smooth l1 forward
    errors.device(*place) = diff.unaryExpr(SmoothL1LossForward<T>(sigma2));

    // multiply outside weight
    if (has_weight) {
      auto outside_weight = EigenVector<T>::Flatten(*in3);
      errors.device(*place) = errors * outside_weight;
    }
    auto loss = EigenVector<T>::Flatten(*out1);
    // first dimension of 'X' is the number of samples
    auto mat_dims =
        phi::make_ddim({static_cast<int>(in0->dims()[0]),
                        static_cast<int>(in_counts / in0->dims()[0])});
    auto errors_mat_view = EigenMatrix<T>::From(ptensor_errors, mat_dims);
    loss.device(*place) = errors_mat_view.sum(Eigen::array<int, 1>({{1}}));
  }
};

template <typename T>
struct SmoothL1LossBackward {
  HOSTDEVICE SmoothL1LossBackward(const T& sigma2) : sigma2(sigma2) {}

  HOSTDEVICE T operator()(const T& val) const {
    T abs_val = std::abs(val);
    if (abs_val < 1.0 / sigma2) {
      return sigma2 * val;
    } else {
      return (0 < val) - (val < 0);
    }
  }

  T sigma2;
};

template <typename DeviceContext, typename T, typename AttrType = T>
class SmoothL1LossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<phi::DenseTensor>("InsideWeight");
    auto* in1 = context.Input<phi::DenseTensor>("OutsideWeight");
    auto* in2 = context.Input<phi::DenseTensor>("Diff");
    auto* og = context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto sigma = static_cast<T>(context.Attr<AttrType>("sigma"));
    T sigma2 = sigma * sigma;
    bool has_weight = (in0 != nullptr) && (in1 != nullptr);

    auto* place =
        context.template device_context<DeviceContext>().eigen_device();

    auto in_dims = in2->dims();
    auto counts = in2->numel();
    auto cols = counts / in_dims[0];
    auto mat_dims =
        phi::make_ddim({static_cast<int>(in_dims[0]), static_cast<int>(cols)});

    Tensor ptensor_diff;
    ptensor_diff.mutable_data<T>({static_cast<int>(counts)},
                                 context.GetPlace());
    auto diff = EigenVector<T>::Flatten(ptensor_diff);
    // apply smooth l1 backwoard
    diff.device(*place) = EigenVector<T>::Flatten(*in2).unaryExpr(
        SmoothL1LossBackward<T>(sigma2));

    // compute weights
    Tensor ptensor_weights;
    ptensor_weights.mutable_data<T>(mat_dims, context.GetPlace());
    auto weights = EigenMatrix<T>::From(ptensor_weights);
    // initialize to 1.0
    weights.device(*place) = weights.constant(static_cast<T>(1.0));
    if (has_weight) {
      auto inside_weight = EigenMatrix<T>::From(*in0, mat_dims);
      auto outside_weight = EigenMatrix<T>::From(*in1, mat_dims);
      weights.device(*place) = inside_weight * outside_weight;
    }

    // compute gradients
    auto out_grad = EigenMatrix<T>::From(*og);
    auto diff_mat_view = EigenMatrix<T>::From(ptensor_diff, mat_dims);
    auto gradients = out_grad.broadcast(
                         Eigen::array<int, 2>({{1, static_cast<int>(cols)}})) *
                     weights * diff_mat_view;

    auto* out0 = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* out1 = context.Output<phi::DenseTensor>(framework::GradVarName("Y"));

    if (out0) {
      out0->mutable_data<T>(context.GetPlace());
      auto x_grad = EigenMatrix<T>::From(*out0, mat_dims);
      x_grad.device(*place) = gradients;
    }

    if (out1) {
      out1->mutable_data<T>(context.GetPlace());
      auto y_grad = EigenMatrix<T>::From(*out1, mat_dims);
      y_grad.device(*place) = -1 * gradients;
    }
  }
};

}  // namespace operators
}  // namespace paddle
