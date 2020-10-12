/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include <random>
#include "paddle/fluid/memory/memcpy.h"

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/fill_constant_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class XPUGaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.Attr<float>("mean");
    float std = context.Attr<float>("std");
    auto* tensor = context.Output<framework::Tensor>("Out");

    std::normal_distribution<T> dist(mean, std);
    auto shape = GetShape(context);
    tensor->Resize(shape);
    int64_t size = tensor->numel();
    T* data = tensor->mutable_data<T>(context.GetPlace());
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    auto engine = framework::GetCPURandomEngine(seed);

    T* data_host = reinterpret_cast<T*>(std::malloc(size * sizeof(T)));
    for (int64_t i = 0; i < size; ++i) {
      data_host[i] = dist(*engine);
    }

    if (std::is_same<T, float>::value &&
        std::is_same<DeviceContext, platform::XPUDeviceContext>::value) {
      platform::XPUPlace place =
          BOOST_GET_CONST(platform::XPUPlace, context.GetPlace());
      memory::Copy(place, data, platform::CPUPlace(), data_host,
                   size * sizeof(T));
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "Unsupported place! Only support XPU device."));
    }

    std::free(data_host);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(gaussian_random,
                       paddle::operators::XPUGaussianRandomKernel<
                           paddle::platform::XPUDeviceContext, float>);
#endif
