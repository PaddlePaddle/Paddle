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

#include <vector>
#include "paddle/framework/ddim.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    auto out_stride = framework::stride_numel(out->dims());
    int64_t before = out_stride[0] / out_stride[axis];
    int64_t out_after = out_stride[axis];

    size_t output_offset = 0;
    for (auto* in : ins) {
      auto in_stride = framework::stride_numel(in->dims());
      int64_t in_after = in_stride[axis];
      for (int64_t i = 0; i < before; ++i) {
        if (platform::is_cpu_place(place)) {
          auto& cpu_place = boost::get<platform::CPUPlace>(place);
          memory::Copy(
              cpu_place, out->data<T>() + output_offset + i * out_after,
              cpu_place, in->data<T>() + i * in_after, sizeof(T) * in_after);
        } else {
#ifdef PADDLE_WITH_CUDA
          auto& gpu_place = boost::get<platform::CUDAPlace>(place);
          auto& cuda_ctx =
              reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx);
          memory::Copy(gpu_place, out->data<T>() +
                       output_offset + i * out_after,
                       gpu_place, in->data<T>() + i * in_after,
                       sizeof(T) * in_after, cuda_ctx.stream()));
#else
          PADDLE_THROW("Paddle is not compiled with GPU");
#endif
        }
      }
      output_offset += in_after;
    }
  }
};

template <typename DeviceContext, typename T>
class ConcatGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto outs = ctx.MultiOutput<framework::Tensor>(framework::GradVarName("X"));
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    size_t input_offset = 0;
    auto in_stride = framework::stride_numel(in->dims());
    auto place = ctx.GetPlace();

    // numel before the specified axis
    int64_t before = in_stride[0] / in_stride[axis];
    int64_t in_after = in_stride[axis];
    for (auto& out : outs) {
      out->mutable_data<T>(ctx.GetPlace());
      auto out_stride = framework::stride_numel(out->dims());
      int64_t out_after = out_stride[axis];
      for (int64_t i = 0; i < before; ++i) {
        if (platform::is_cpu_place(place)) {
          auto& cpu_place = boost::get<platform::CPUPlace>(place);
          memory::Copy(cpu_place, out->data<T>() + i * out_after, cpu_place,
                       in->data<T>() + input_offset + i * in_after,
                       sizeof(T) * out_after);
        } else {
#ifdef PADDLE_WITH_CUDA
          auto& gpu_place = boost::get<platform::CUDAPlace>(place);
          auto& cuda_ctx =
              reinterpret_cast<const platform::CUDADeviceContext&>(dev_ctx);
          memory::Copy(gpu_place, out->data<T>() + i * out_after, gpu_place,
                       in->data<T>() + input_offset + i * in_after,
                       sizeof(T) * out_after, cuda_ctx.stream());
#else
          PADDLE_THROW("Paddle is not compiled with GPU");
#endif
        }
      }
      input_offset += out_after;
    }
  }
};

}  // namespace operators
}  // namespace paddle
