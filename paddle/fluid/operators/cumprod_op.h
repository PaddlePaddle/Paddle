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

#include <cstdint>
#include <type_traits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

static void GetCumprodDimInfo(const framework::DDim& dim, int cumprod_dim,
                              size_t* outer_dim, size_t* mid_dim,
                              size_t* inner_dim) {
  PADDLE_ENFORCE_GE(
      cumprod_dim, -dim.size(),
      platform::errors::InvalidArgument(
          "The input dim of CumprodOp should be larger than the opposite "
          "rank of input x which is %d.But received dim=%d",
          -dim.size(), cumprod_dim));
  PADDLE_ENFORCE_LT(cumprod_dim, dim.size(),
                    platform::errors::InvalidArgument(
                        "The input dim of CumprodOp should be smaller than the "
                        "rank of input x which is %d.But received dim=%d",
                        dim.size(), cumprod_dim));
  if (cumprod_dim < 0) cumprod_dim += dim.size();

  *outer_dim = 1;
  for (int i = 0; i < cumprod_dim; ++i) {
    *outer_dim *= dim[i];
  }
  *mid_dim = dim[cumprod_dim];
  *inner_dim = 1;
  for (int i = cumprod_dim + 1; i < dim.size(); ++i) {
    *inner_dim *= dim[i];
  }
}

template <typename T>
class CumprodOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    int dim = context.Attr<int>("dim");

    auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    framework::DDim shape = x->dims();

    size_t outer_dim = 1;
    size_t mid_dim = 1;
    size_t inner_dim = 1;
    GetCumprodDimInfo(shape, dim, &outer_dim, &mid_dim, &inner_dim);

    for (size_t i = 0; i < outer_dim; i++) {
      for (size_t j = 0; j < mid_dim; j++) {
        for (size_t k = 0; k < inner_dim; k++) {
          size_t pos = i * mid_dim * inner_dim + j * inner_dim + k;
          if (j == 0) {
            out_data[pos] = x_data[pos];
          } else {
            out_data[pos] = out_data[pos - inner_dim] * x_data[pos];
          }
        }
      }
    }
  }
};

template <typename T>
class CumprodGradOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const {
    const Tensor* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* out = context.Input<Tensor>("Out");

    int dim = context.Attr<int>("dim");
    framework::DDim shape = x->dims();
    Tensor* d_x = context.Output<Tensor>(framework::GradVarName("X"));

    auto* d_out_data = d_out->data<T>();
    auto* x_data = x->data<T>();
    auto* out_data = out->data<T>();
    auto* d_x_data = d_x->mutable_data<T>(context.GetPlace());

    auto place = context.GetPlace();
    const auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();

    size_t outer_dim = 1;
    size_t mid_dim = 1;
    size_t inner_dim = 1;
    GetCumprodDimInfo(shape, dim, &outer_dim, &mid_dim, &inner_dim);
    size_t numel = outer_dim * mid_dim * inner_dim;

    // deal with complex
    const T* x_data_deal;
    const T* out_data_deal;
    memory::AllocationPtr x_conj;
    memory::AllocationPtr out_conj;
    if (framework::IsComplex<T>::value) {
      x_conj = memory::Alloc(place, numel * sizeof(T));
      auto* x_data_conj = reinterpret_cast<T*>(x_conj->ptr());
      out_conj = memory::Alloc(place, numel * sizeof(T));
      auto* out_data_conj = reinterpret_cast<T*>(out_conj->ptr());

      platform::ForRange<platform::CPUDeviceContext> for_range_x(dev_ctx,
                                                                 numel);
      math::ConjFunctor<T> functor_x(x_data, numel, x_data_conj);
      for_range_x(functor_x);

      platform::ForRange<platform::CPUDeviceContext> for_range_out(dev_ctx,
                                                                   numel);
      math::ConjFunctor<T> functor_out(out_data, numel, out_data_conj);
      for_range_out(functor_out);

      x_data_deal = x_data_conj;
      out_data_deal = out_data_conj;
    } else {
      x_data_deal = x_data;
      out_data_deal = out_data;
    }

    for (size_t i = 0; i < outer_dim; i++) {
      for (size_t k = 0; k < inner_dim; k++) {
        for (size_t j = 0; j < mid_dim; j++) {
          size_t index = i * mid_dim * inner_dim + j * inner_dim + k;
          d_x_data[index] = 0;
          for (size_t n = 0; n < mid_dim; n++) {
            size_t pos = i * mid_dim * inner_dim + n * inner_dim + k;
            T elem;
            if (j == 0) {
              elem = d_out_data[pos];
            } else {
              elem = d_out_data[pos] * out_data_deal[index - inner_dim];
            }
            if (pos > index) {
              for (size_t m = index + inner_dim; m <= pos; m += inner_dim) {
                elem *= x_data_deal[m];
              }
            } else if (pos < index) {
              elem = static_cast<T>(0);
            }
            d_x_data[index] += elem;
          }
        }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
