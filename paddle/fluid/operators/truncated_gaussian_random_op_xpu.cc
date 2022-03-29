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

#include "paddle/fluid/operators/truncated_gaussian_random_op.h"
#include <limits>
#include <random>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class XPUTruncatedGaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.Attr<float>("mean");
    float std = context.Attr<float>("std");
    auto* tensor = context.Output<framework::Tensor>("Out");
    T* data = tensor->mutable_data<T>(context.GetPlace());

    std::uniform_real_distribution<T> dist(std::numeric_limits<float>::min(),
                                           1.0);
    TruncatedNormal<T> truncated_normal(mean, std);
    int64_t size = tensor->numel();

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    // TODO(pangyoki): implement GetXPURandomEngine to set different seeds on
    // corresponding XPU device.
    auto engine = framework::GetCPURandomEngine(seed);

    std::unique_ptr<T[]> data_cpu(new T[size]);

    for (int64_t i = 0; i < size; ++i) {
      data_cpu[i] = truncated_normal(dist(*engine));
    }

    memory::Copy(context.GetPlace(), data, platform::CPUPlace(),
                 reinterpret_cast<void*>(data_cpu.get()), size * sizeof(T));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(truncated_gaussian_random,
                       ops::XPUTruncatedGaussianRandomKernel<
                           paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
