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
#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

template <typename DeviceContext, typename T>
class GroupNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* x = ctx.Input<Tensor>("X");

    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* var = ctx.Output<Tensor>("Variance");
    const auto groups = ctx.Attr<int>("groups");

    const auto x_dims = x->dims();
    const int group_size = (x_dims[1] - 1) / groups + 1;

    y->mutable_data<T>(ctx.GetPlace());
    mean->mutable_data<T>(ctx.GetPlace());
    var->mutable_data<T>(ctx.GetPlace());

    auto* x_data = x->data<T>();
    auto* y_data = y->data<T>();
    auto* mean_data = mean->data<T>();
    auto* var_data = var->data<T>();

    const T* scale_data = nullptr;
    if (scale) scale_data = scale->data<T>();
    const T* bias_data = nullptr;
    if (bias) bias_data = bias->data<T>();

    int imsize = x_dims[2] * x_dims[3];
    auto* iter_x_data = x_data;
    auto* iter_y_data = y_data;
    for (int bid = 0; bid < x_dims[0]; bid++)
      for (int gid = 0; gid < groups; gid++) {
        T x_mean = 0, x_var = 0;
        int number = std::min(group_size,
                              static_cast<int>(x_dims[1] - gid * group_size));
        auto* tmp = iter_x_data;
        for (int cid = 0; cid < number; cid++) {
          for (int imid = 0; imid < imsize; imid++, iter_x_data++) {
            x_mean += iter_x_data[0];
            x_var += iter_x_data[0] * iter_x_data[0];
          }
        }
        x_mean /= number * imsize;
        x_var /= number * imsize;
        x_var = x_var - x_mean * x_mean;
        T var_inv = 1.0 / sqrt(x_var + epsilon);
        mean_data[bid * groups + gid] = x_mean;
        var_data[bid * groups + gid] = x_var;
        for (int cid = 0; cid < number; cid++) {
          for (int imid = 0; imid < imsize; imid++, tmp++, iter_y_data++) {
            T val = (tmp[0] - x_mean) * var_inv;
            if (scale_data) val *= scale_data[gid * group_size + cid];
            if (bias_data) val += bias_data[gid * group_size + cid];
            iter_y_data[0] = val;
          }
        }
      }
  }
};

template <typename DeviceContext, typename T>
class GroupNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* x = ctx.Input<Tensor>("X");
    auto* mean = ctx.Input<Tensor>("Mean");
    auto* var = ctx.Input<Tensor>("Variance");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto groups = ctx.Attr<int>("groups");

    // init output
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto& x_dims = x->dims();
    const int group_size = (x_dims[1] - 1) / groups + 1;

    // TODO(liangdun): need to check d_x is null
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    T* d_x_data = nullptr;
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, d_x, static_cast<T>(0));
      d_x_data = d_x->data<T>();
    }

    auto* x_data = x->data<T>();
    auto* y_data = d_y->data<T>();
    auto* mean_data = mean->data<T>();
    auto* var_data = var->data<T>();
    T* d_scale_data = nullptr;
    if (d_scale) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, d_scale, static_cast<T>(0));
      d_scale_data = d_scale->data<T>();
    }
    T* d_bias_data = nullptr;
    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, d_bias, static_cast<T>(0));
      d_bias_data = d_bias->data<T>();
    }

    const T* scale_data = nullptr;
    if (scale) scale_data = scale->data<T>();

    int imsize = x_dims[2] * x_dims[3];
    auto* iter_x_data = x_data;
    auto* iter_d_x_data = d_x_data;
    auto* iter_y_data = y_data;
    for (int bid = 0; bid < x_dims[0]; bid++)
      for (int gid = 0; gid < groups; gid++) {
        T x_mean = mean_data[bid * groups + gid];
        T x_var = var_data[bid * groups + gid];
        T var_inv = 1.0 / sqrt(x_var + epsilon);
        int number = std::min(group_size,
                              static_cast<int>(x_dims[1] - gid * group_size));
        auto* tmp = iter_x_data;
        auto* tmp2 = iter_d_x_data;
        T d_var_inv = 0, d_x_mean = 0;
        for (int cid = 0; cid < number; cid++) {
          for (int imid = 0; imid < imsize;
               imid++, tmp++, iter_y_data++, iter_d_x_data++) {
            T val = (tmp[0] - x_mean) * var_inv;
            T dval = iter_y_data[0];
            if (d_bias_data) d_bias_data[gid * group_size + cid] += dval;
            if (d_scale_data)
              d_scale_data[gid * group_size + cid] += val * dval;
            if (scale_data) dval = scale_data[gid * group_size + cid] * dval;

            d_var_inv += (tmp[0] - x_mean) * dval;
            T d_tmp = dval * var_inv;
            if (d_x_data) iter_d_x_data[0] += d_tmp;
            d_x_mean -= d_tmp;
          }
        }

        T d_x_var =
            -1.0 / (2 * (x_var + epsilon) * sqrt(x_var + epsilon)) * d_var_inv;
        d_x_mean -= 2 * d_x_var * x_mean;
        d_x_var /= number * imsize;
        d_x_mean /= number * imsize;

        iter_d_x_data = tmp2;

        if (d_x_data) {
          for (int cid = 0; cid < number; cid++) {
            for (int imid = 0; imid < imsize;
                 imid++, iter_x_data++, iter_d_x_data++) {
              iter_d_x_data[0] += d_x_mean;
              iter_d_x_data[0] += iter_x_data[0] * 2 * d_x_var;
            }
          }
        }
      }
  }
};

}  // namespace operators
}  // namespace paddle
