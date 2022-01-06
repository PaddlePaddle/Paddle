// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "math.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"
namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

// template <typename T>
// struct NormDimValueFunctor<T> {
//   NormDimValueFunctor(T* input, T* output, int64_t dim_divisor, int64_t
//   dimension_each, float p)
//       : input_(input), output_(output),dim_divisor_(dim_divisor),
//       dimension_each_(dimension_each),p_(p) {}

//   HOSTDEVICE void operator()(int64_t i) const {
//       auto dim_index = i / dim_divsor % dimension_each;
//       dim_value[dim_index] += std::pow(std::abs(input[i]), p);
//   }

//   T* input_;
//   T* output_;
//   int64_t dimension_each_, dim_divisor_;
//   float p_,max_norm_;

// };
// template <typename DeviceContext, typename T>
template <typename T>
class CPURenormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    auto numel = x->numel();
    auto* x_data = x->data<T>();
    auto input_dims = x->dims();
    float max_norm = context.Attr<float>("max_norm");
    float p = context.Attr<float>("p");
    int dim = context.Attr<int>("axis");
    auto dimension_each = input_dims[dim];
    auto dim_size = input_dims.size();
    int64_t dim_divisor = 1;
    for (int i = dim + 1; i < dim_size; i++) dim_divisor *= input_dims[i];

    // auto& dev_ctx = ctx.template device_context<DeviceContext>();
    // std::vector<int64_t> dim_index(dim_size, 0);
    std::vector<T> dim_value(dimension_each,
                             0);  // dim_value = (x1^p + x2^p + x3^p....)^(1/p)

    auto* out_data =
        out->mutable_data<T>(context.GetPlace(), size_t(numel * sizeof(T)));

    int64_t index = 0, dim_index = 0;
    for (int64_t i = 0; i < numel; i++) {
      // auto dim_index = i / dim_divsor % dimension_each;
      dim_value[dim_index] += std::pow(std::abs(x_data[i]), p);
      index++;
      if (index == dim_divisor) {
        dim_index++;
        if (dim_index == dimension_each) {
          dim_index = 0;
        }
        index = 0;
      }
    }
    for (int64_t i = 0; i < dimension_each; i++) {
      dim_value[i] = std::pow(dim_value[i], 1.0 / p);
      if (dim_value[i] > max_norm)
        dim_value[i] = max_norm / dim_value[i];
      else
        dim_value[i] = 1.0;
      // dim_index[i] = 0;
    }
    index = dim_index = 0;
    for (int64_t i = 0; i < numel; i++) {
      // auto dim_index = i / dim_divsor % dimension_each;
      out_data[i] = dim_value[dim_index] < 1.0
                        ? dim_value[dim_index] * x_data[i]
                        : x_data[i];
      index++;
      if (index == dim_divisor) {
        dim_index++;
        if (dim_index == dimension_each) {
          dim_index = 0;
        }
        index = 0;
      }
    }
  }
};

// template <typename DeviceContext, typename T>
template <typename T>
class CPURenormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    const framework::Tensor* x = ctx.Input<framework::Tensor>("X");
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto numel = d_out->numel();
    auto* dout_data = d_out->data<T>();
    auto* x_data = x->data<T>();
    auto input_dims = x->dims();
    float max_norm = ctx.Attr<float>("max_norm");
    float p = ctx.Attr<float>("p");
    int dim = ctx.Attr<int>("axis");
    auto dimension_each = input_dims[dim];
    auto dim_size = input_dims.size();
    int64_t dim_divisor = 1;
    for (int i = dim + 1; i < dim_size; i++) dim_divisor *= input_dims[i];
    auto* dx_data = d_x->mutable_data<T>(
        ctx.GetPlace(), static_cast<size_t>(numel * sizeof(T)));
    std::vector<T> dim_value(dimension_each, 0),
        dim_power_sum(dimension_each, 0),
        weight_derivative(dimension_each, 0.0);
    int64_t index = 0, dim_index = 0;
    for (int64_t i = 0; i < numel; i++) {
      // auto dim_index = i / dim_divsor % dimension_each;
      dim_value[dim_index] += std::pow(std::abs(x_data[i]), p);
      index++;
      if (index == dim_divisor) {
        dim_index++;
        if (dim_index == dimension_each) {
          dim_index = 0;
        }
        index = 0;
      }
    }
    for (int64_t i = 0; i < dimension_each; i++) {
      auto temp = std::pow(dim_value[i], 1.0 / p);
      if (temp > max_norm) {
        dim_power_sum[i] =
            std::pow(dim_value[i], (T)(-1.0 - 1.0 / p)) * -1 * max_norm;
        dim_value[i] = max_norm / temp;
      } else
        dim_value[i] = 1.0;
    }
    index = dim_index = 0;
    for (int64_t i = 0; i < numel; i++) {
      // auto dim_index = i / dim_divsor % dimension_each;
      dx_data[i] = dim_value[dim_index] * dout_data[i];
      weight_derivative[dim_index] += x_data[i] * dout_data[i];
      index++;
      if (index == dim_divisor) {
        dim_index++;
        if (dim_index == dimension_each) {
          dim_index = 0;
        }
        index = 0;
      }
    }
    index = dim_index = 0;
    for (int64_t i = 0; i < numel; i++) {
      // auto dim_index = i / dim_divsor % dimension_each;
      dx_data[i] += weight_derivative[dim_index] * dim_power_sum[dim_index] *
                    std::pow(std::abs(x_data[i]), p - 1.0) *
                    (x_data[i] >= 0 ? 1 : -1);
      index++;
      if (index == dim_divisor) {
        dim_index++;
        if (dim_index == dimension_each) {
          dim_index = 0;
        }
        index = 0;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
