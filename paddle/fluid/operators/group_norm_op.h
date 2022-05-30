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
#include <array>
#include <numeric>
#include <string>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

template <typename DeviceContext, typename T>
class GroupNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* x = ctx.Input<Tensor>("Y");
    auto* var = ctx.Input<Tensor>("Variance");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto groups = ctx.Attr<int>("groups");

    // init output
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto& x_dims = x->dims();
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int group_size = C / groups;

    d_x->mutable_data<T>(ctx.GetPlace());
    phi::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    auto* x_data = x->data<T>();
    auto* d_x_data = d_x->data<T>();
    auto* y_data = d_y->data<T>();
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
    const T* bias_data = nullptr;
    if (bias) bias_data = bias->data<T>();

    int imsize = 1;
    if (data_layout == DataLayout::kNCHW) {
      for (int i = 2; i < x_dims.size(); ++i) {
        imsize *= x_dims[i];
      }
    } else {
      for (int i = 1; i < x_dims.size() - 1; ++i) {
        imsize *= x_dims[i];
      }
    }
    auto* iter_x_data = x_data;
    auto* iter_d_x_data = d_x_data;
    auto* iter_y_data = y_data;
    for (int bid = 0; bid < x_dims[0]; bid++) {
      for (int gid = 0; gid < groups; gid++) {
        T x_var = var_data[bid * groups + gid];
        T var_inv = 1.0 / sqrt(x_var + epsilon);
        int number =
            std::min(group_size, static_cast<int>(C - gid * group_size));
        T number_inv = 1.0 / (number * imsize);
        auto* tmp_x = iter_x_data;
        auto* tmp_y = iter_y_data;
        auto* tmp_d_x = iter_d_x_data;
        auto* x_src_data = iter_x_data;
        auto* y_src_data = iter_y_data;
        auto* iter_x_data_backup = iter_x_data;
        auto* iter_y_data_backup = iter_y_data;
        auto* iter_d_x_data_backup = iter_d_x_data;
        T dp_scale = 0, dp_bias = 0;

        if (data_layout == DataLayout::kNCHW) {
          for (int cid = 0; cid < number; cid++) {
            for (int imid = 0; imid < imsize;
                 imid++, iter_x_data++, iter_y_data++) {
              T val = iter_x_data[0];
              if (bias_data) val -= bias_data[gid * group_size + cid];
              T dval = iter_y_data[0];
              dp_scale += val * dval;
              if (scale_data)
                dp_bias += dval * scale_data[gid * group_size + cid];

              if (scale_data && scale_data[gid * group_size + cid] != 0)
                val /= scale_data[gid * group_size + cid];
              if (d_bias_data) d_bias_data[gid * group_size + cid] += dval;
              if (d_scale_data)
                d_scale_data[gid * group_size + cid] += val * dval;
            }
          }

          for (int cid = 0; cid < number; cid++) {
            for (int imid = 0; imid < imsize;
                 imid++, iter_d_x_data++, tmp_x++, tmp_y++) {
              T v_y = tmp_x[0];
              T dly = tmp_y[0];
              T dss = dp_scale;
              T dbs = dp_bias;
              T v_scale = 1., v_bias = 0.;
              if (scale_data) v_scale = scale_data[gid * group_size + cid];
              if (bias_data) v_bias = bias_data[gid * group_size + cid];
              v_y -= v_bias;
              if (v_scale != 0) v_y /= v_scale;
              iter_d_x_data[0] =
                  (dly * v_scale - number_inv * dss * v_y - number_inv * dbs) *
                  var_inv;
            }
          }
        } else {
          for (int cid = 0; cid < number; cid++) {
            iter_x_data = x_src_data + cid;
            iter_y_data = y_src_data + cid;
            for (int imid = 0; imid < imsize;
                 imid++, iter_x_data += C, iter_y_data += C) {
              T val = iter_x_data[0];
              if (bias_data) val -= bias_data[gid * group_size + cid];
              T dval = iter_y_data[0];
              dp_scale += val * dval;
              if (scale_data)
                dp_bias += dval * scale_data[gid * group_size + cid];

              if (scale_data && scale_data[gid * group_size + cid] != 0)
                val /= scale_data[gid * group_size + cid];
              if (d_bias_data) d_bias_data[gid * group_size + cid] += dval;
              if (d_scale_data)
                d_scale_data[gid * group_size + cid] += val * dval;
            }
          }

          for (int cid = 0; cid < number; cid++) {
            tmp_x = x_src_data + cid;
            tmp_y = y_src_data + cid;
            iter_d_x_data = tmp_d_x + cid;
            for (int imid = 0; imid < imsize;
                 imid++, iter_d_x_data += C, tmp_x += C, tmp_y += C) {
              T v_y = tmp_x[0];
              T dly = tmp_y[0];
              T dss = dp_scale;
              T dbs = dp_bias;
              T v_scale = 1.0, v_bias = 0.;
              if (scale_data) v_scale = scale_data[gid * group_size + cid];
              if (bias_data) v_bias = bias_data[gid * group_size + cid];
              v_y -= v_bias;
              if (v_scale != 0) v_y /= v_scale;
              iter_d_x_data[0] =
                  (dly * v_scale - number_inv * dss * v_y - number_inv * dbs) *
                  var_inv;
            }
          }
          iter_x_data = iter_x_data_backup + group_size;
          iter_y_data = iter_y_data_backup + group_size;
          iter_d_x_data = iter_d_x_data_backup + group_size;
        }
      }
      if (data_layout == DataLayout::kNHWC) {
        iter_x_data = x_data + (bid + 1) * C * imsize;
        iter_d_x_data = d_x_data + (bid + 1) * C * imsize;
        iter_y_data = y_data + (bid + 1) * C * imsize;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
