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
#include "paddle/fluid/memory/memory.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class FakeDequantizeMaxAbsKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(in->place());

    int num_bits = ctx.Attr<int>("num_bits");
    int range = std::pow(2, num_bits - 1) - 1;

    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();

    T s;
    auto place = ctx.GetPlace();
    if (platform::is_cpu_place(place)) {
      s = scale->data<T>()[0];
    } else {
#ifdef PADDLE_WITH_CUDA
      auto& gpu_place = boost::get<platform::CUDAPlace>(place);
      auto stream =
          ctx.template device_context<platform::CUDADeviceContext>().stream();
      memory::Copy(platform::CPUPlace(), &s, gpu_place, scale->data<T>(),
                   sizeof(T), stream);
#else
      PADDLE_THROW("Paddle is not compiled with GPU");
#endif
    }

    eigen_out.device(dev) = (s / range) * eigen_in;
  }
};

}  // namespace operators
}  // namespace paddle
