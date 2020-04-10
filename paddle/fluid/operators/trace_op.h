/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
void Trace(framework::DDim input_dims, framework::DDim input_stride,
           framework::DDim output_stride, int64_t numel, const T* input_data,
           T* out_data, int64_t offset, int64_t dim1, int64_t dim2,
           bool is_grad) {
  auto dim1_ = dim1 < 0 ? input_dims.size() + dim1 : dim1;
  auto dim2_ = dim2 < 0 ? input_dims.size() + dim2 : dim2;
  auto len1 = input_dims[std::min(dim1_, dim2_)];
  auto len2 = input_dims[std::max(dim1_, dim2_)];
  auto stride1 = input_stride[std::min(dim1_, dim2_)];
  auto stride2 = input_stride[std::max(dim1_, dim2_)];

  int offset_stride = 0;
  if (offset >= 0) {
    offset_stride = stride2;
    len2 -= offset;
  } else {
    offset_stride = stride1;
    len1 += offset;
  }
  int diag_size = len2 < len1 ? len2 : len1;

  auto ret_strides = vectorize(input_stride);
  ret_strides.erase(ret_strides.begin() + std::max(dim1_, dim2_));
  ret_strides.erase(ret_strides.begin() + std::min(dim1_, dim2_));

  if (diag_size > 0) {
    int pos = std::abs(offset) * offset_stride;
    for (int idx = 0; idx < numel; idx++) {
      int position = pos;
      int64_t num = idx;
      for (size_t i = 0; i < ret_strides.size(); i++) {
        position += num / output_stride[i] * ret_strides[i];
        num = num % output_stride[i];
      }
      for (int j = 0; j < diag_size; j++) {
        if (is_grad) {
          out_data[position] = input_data[idx];
        } else {
          out_data[idx] += input_data[position];
        }
        position += stride1 + stride2;
      }
    }
  }
}

template <typename DeviceContext, typename T>
class TraceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* input = context.Input<framework::Tensor>("Input");
    auto* out = context.Output<framework::Tensor>("Out");

    int64_t offset = context.Attr<int>("offset");
    int64_t dim1 = context.Attr<int>("dim1");
    int64_t dim2 = context.Attr<int>("dim2");

    auto input_dims = input->dims();
    auto input_stride = framework::stride(input_dims);
    auto output_dims = out->dims();
    auto output_stride = framework::stride(output_dims);
    auto numel = out->numel();

    auto* input_data = input->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, out, static_cast<T>(0.0));

    Trace<DeviceContext, T>(input_dims, input_stride, output_stride, numel,
                            input_data, out_data, offset, dim1, dim2, false);
  }
};

template <typename DeviceContext, typename T>
class TraceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_x =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    int64_t offset = context.Attr<int>("offset");
    int64_t dim1 = context.Attr<int>("dim1");
    int64_t dim2 = context.Attr<int>("dim2");

    auto input_dims = d_x->dims();
    auto input_stride = framework::stride(input_dims);
    auto output_dims = d_out->dims();
    auto output_stride = framework::stride(output_dims);
    auto numel = d_out->numel();

    auto* input_data = d_out->data<T>();
    T* out_data = d_x->mutable_data<T>(context.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, d_x, static_cast<T>(0.0));

    Trace<DeviceContext, T>(input_dims, input_stride, output_stride, numel,
                            input_data, out_data, offset, dim1, dim2, true);
  }
};

}  // namespace operators
}  // namespace paddle
