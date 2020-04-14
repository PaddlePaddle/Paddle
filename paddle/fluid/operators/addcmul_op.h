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
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class AddcmulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input = ctx.Input<framework::Tensor>("Input");
    auto *tensor1 = ctx.Input<framework::Tensor>("Tensor1");
    auto *tensor2 = ctx.Input<framework::Tensor>("Tensor2");
    auto tensor1_dims = tensor1->dims();
    auto tensor2_dims = tensor2->dims();
    int max_dim = std::max(tensor1_dims.size(), tensor2_dims.size());
    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(tensor1_dims.size() - tensor2_dims.size())
                       : axis);
    std::vector<int> tensor1_dims_array(max_dim);
    std::vector<int> tensor2_dims_array(max_dim);
    std::vector<int> muled_dims_array(max_dim);
    GetBroadcastDimsArrays(tensor1_dims, tensor2_dims,
                           tensor1_dims_array.data(), tensor2_dims_array.data(),
                           muled_dims_array.data(), max_dim, axis);
    framework::Tensor muled;
    muled.mutable_data<T>(framework::make_ddim(muled_dims_array),
                          ctx.GetPlace());
    auto dims_equal = tensor1->dims() == tensor2->dims();
    if (dims_equal) {
      SameDimsElemwiseMul<DeviceContext, T> same_dims_mul;
      same_dims_mul(ctx, tensor1, tensor2, &muled);
    } else {
      default_elementwise_mul<DeviceContext, T>(ctx, tensor1, tensor2, &muled);
    }

    T value = static_cast<T>(ctx.Attr<float>("value"));
    auto muled_elems = muled.numel();
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    blas.SCAL(muled_elems, value, muled.data<T>());

    auto *out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    dims_equal = input->dims() == muled.dims();
    if (dims_equal) {
      SameDimsElemwiseAdd<DeviceContext, T> same_dims_add;
      same_dims_add(ctx, input, &muled, out);
    } else {
      default_elementwise_add<DeviceContext, T>(ctx, input, &muled, out);
    }
  }
};

template <typename DeviceContext, typename T>
class AddcmulGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto *input = ctx.Input<Tensor>("Input");
    auto *tensor1 = ctx.Input<Tensor>("Tensor1");
    auto *tensor2 = ctx.Input<Tensor>("Tensor2");
    auto *d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *out = d_out;
    auto *d_input = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto *d_tensor1 = ctx.Output<Tensor>(framework::GradVarName("Tensor1"));
    auto *d_tensor2 = ctx.Output<Tensor>(framework::GradVarName("Tensor2"));
    int axis = ctx.Attr<int>("axis");

    auto tensor1_dims = tensor1->dims();
    auto tensor2_dims = tensor2->dims();
    int max_dim = std::max(tensor1_dims.size(), tensor2_dims.size());
    axis = (axis == -1 ? std::abs(tensor1_dims.size() - tensor2_dims.size())
                       : axis);
    std::vector<int> tensor1_dims_array(max_dim);
    std::vector<int> tensor2_dims_array(max_dim);
    std::vector<int> muled_dims_array(max_dim);
    GetBroadcastDimsArrays(tensor1_dims, tensor2_dims,
                           tensor1_dims_array.data(), tensor2_dims_array.data(),
                           muled_dims_array.data(), max_dim, axis);

    framework::Tensor muled, d_muled;
    auto muled_dims = framework::make_ddim(muled_dims_array);
    muled.mutable_data<T>(muled_dims, ctx.GetPlace());
    d_muled.mutable_data<T>(muled_dims, ctx.GetPlace());

    if (d_input != nullptr && (d_input->dims() == muled_dims)) {
      elementwise_add_grad<DeviceContext, T>(ctx, input, &muled, out, d_out,
                                             d_input, &d_muled);
    } else {
      ElemwiseExplicitGradCompute<DeviceContext, T, IdentityGrad<T>,
                                  IdentityGrad<T>>(
          ctx, *input, muled, *out, *d_out, axis, d_input, &d_muled,
          IdentityGrad<T>(), IdentityGrad<T>());
    }

    axis = ctx.Attr<int>("axis");
    if (d_tensor1 != nullptr && d_tensor2 != nullptr &&
        (d_tensor1->dims() == d_tensor2->dims())) {
      elementwise_mul_grad<DeviceContext, T>(ctx, tensor1, tensor2, &muled,
                                             &d_muled, d_tensor1, d_tensor2);
    } else {
      ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
          ctx, *tensor1, *tensor2, muled, d_muled, axis, d_tensor1, d_tensor2,
          MulGradDX<T>(), MulGradDY<T>());
    }
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    auto tensor1_elems = d_tensor1->numel();
    auto tensor2_elems = d_tensor2->numel();
    T value = static_cast<T>(ctx.Attr<float>("value"));
    blas.SCAL(tensor1_elems, value, d_tensor1->data<T>());
    blas.SCAL(tensor2_elems, value, d_tensor2->data<T>());
  }
};

}  // namespace operators
}  // namespace paddle
