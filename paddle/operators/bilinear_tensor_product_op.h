/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   You may not use this file except in compliance with the License.
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
#include "paddle/platform/transform.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using platform::Transform;

template <typename Place, typename T>
class BilinearTensorProductKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto weight_dims = weight->dims();
    Tensor left_mul_vec;
    left_mul_vec.mutable_data<T>(framework::make_ddim({weight_dims[2]}),
                                 ctx.GetPlace());
    if (bias) {
      out->CopyFrom(*bias, ctx.GetPlace(), ctx.device_context());
    }
    for (int i = 0; i < weight_dims[0]; ++i) {
      Tensor weight_mat = weight->Slice(i, i + 1).Resize(
          framework::make_ddim({weight_dims[1], weight_dims[2]}));
      math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasNoTrans, 1,
                           weight_dims[2], weight_dims[1], 1, x->data<T>(),
                           weight_mat.data<T>(), 0, left_mul_vec.data<T>());
      if (bias) {
        math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasNoTrans,
                             1, 1, weight_dims[2], 1, left_mul_vec.data<T>(),
                             y->data<T>(), 1, &(out->data<T>()[i]));
      } else {
        math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasNoTrans,
                             1, 1, weight_dims[2], 1, left_mul_vec.data<T>(),
                             y->data<T>(), 0, &(out->data<T>()[i]));
      }
    }
  }
};

template <typename T>
class ScaleFunctor {
 public:
  explicit ScaleFunctor(const T* scale) : scale_(scale) {}

  HOSTDEVICE T operator()(const T& x) const { return x * (*scale_); }

 private:
  const T* scale_;
};

template <typename Place, typename T>
class BilinearTensorProductGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* y = ctx.Input<Tensor>("Y");
    const Tensor* weight = ctx.Input<Tensor>("Weight");
    Tensor* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    Tensor* d_y = ctx.Output<Tensor>(framework::GradVarName("Y"));
    Tensor* d_weight = ctx.Output<Tensor>(framework::GradVarName("Weight"));
    Tensor* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    const Tensor* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_out_ptr = d_out->data<T>();
    auto weight_dims = weight->dims();

    // Get the first matrix of Weight.
    Tensor weight_mat_0 = weight->Slice(0, 1).Resize(
        framework::make_ddim({weight_dims[1], weight_dims[2]}));

    // Create the intermediate variable for gradient.
    int numel_x = x->numel();
    int numel_y = y->numel();
    const T* x_ptr = x->data<T>();
    const T* y_ptr = y->data<T>();
    Tensor x_scale;
    T* x_scale_ptr = x_scale.mutable_data<T>(
        framework::make_ddim({weight_dims[1]}), ctx.GetPlace());
    Tensor y_scale;
    T* y_scale_ptr = y_scale.mutable_data<T>(
        framework::make_ddim({weight_dims[2]}), ctx.GetPlace());
    Transform<Place> trans;

    // Caculate the gradient of X according to the first matrix of Weight.
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      trans(ctx.device_context(), y_ptr, y_ptr + numel_y, y_scale_ptr,
            ScaleFunctor<T>(&d_out_ptr[0]));
      math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasTrans, 1,
                           weight_dims[1], weight_dims[2], 1, y_scale.data<T>(),
                           weight_mat_0.data<T>(), 0, d_x->data<T>());
    }

    // Caculate the gradient of Y according to the first matrix of Weight.
    if (d_y) {
      d_y->mutable_data<T>(ctx.GetPlace());
      trans(ctx.device_context(), x_ptr, x_ptr + numel_x, x_scale_ptr,
            ScaleFunctor<T>(&d_out_ptr[0]));
      math::gemm<Place, T>(ctx.device_context(), CblasTrans, CblasNoTrans,
                           weight_dims[2], 1, weight_dims[1], 1,
                           weight_mat_0.data<T>(), x_scale.data<T>(), 0,
                           d_y->data<T>());
    }

    // Caculate the gradient of X and Y completly.
    if (d_x || d_y) {
      for (int i = 1; i < weight_dims[0]; ++i) {
        Tensor weight_mat = weight->Slice(i, i + 1).Resize(
            framework::make_ddim({weight_dims[1], weight_dims[2]}));
        if (d_x) {
          trans(ctx.device_context(), y_ptr, y_ptr + numel_y, y_scale_ptr,
                ScaleFunctor<T>(&d_out_ptr[i]));
          math::gemm<Place, T>(ctx.device_context(), CblasNoTrans, CblasTrans,
                               1, weight_dims[1], weight_dims[2], 1,
                               y_scale.data<T>(), weight_mat.data<T>(), 1,
                               d_x->data<T>());
        }
        if (d_y) {
          trans(ctx.device_context(), x_ptr, x_ptr + numel_x, x_scale_ptr,
                ScaleFunctor<T>(&d_out_ptr[i]));
          math::gemm<Place, T>(ctx.device_context(), CblasTrans, CblasNoTrans,
                               weight_dims[2], 1, weight_dims[1], 1,
                               weight_mat.data<T>(), x_scale.data<T>(), 1,
                               d_y->data<T>());
        }
      }
    }

    // Caculate the gradient of Weight.
    if (d_weight) {
      d_weight->mutable_data<T>(ctx.GetPlace());
      for (int i = 0; i < weight_dims[0]; ++i) {
        Tensor d_weight_mat = d_weight->Slice(i, i + 1).Resize(
            framework::make_ddim({weight_dims[1], weight_dims[2]}));
        trans(ctx.device_context(), x_ptr, x_ptr + numel_x, x_scale_ptr,
              ScaleFunctor<T>(&d_out_ptr[i]));
        math::gemm<Place, T>(ctx.device_context(), CblasTrans, CblasNoTrans,
                             weight_dims[1], weight_dims[2], 1, 1,
                             x_scale.data<T>(), y->data<T>(), 0,
                             d_weight_mat.data<T>());
      }
    }

    // Caculate the gradient of Bias.
    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      d_bias->CopyFrom(*d_out, ctx.GetPlace(), ctx.device_context());
    }
  }
};

}  // namespace operators
}  // namespace paddle
