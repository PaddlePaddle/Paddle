/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <limits>
#include <random>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/truncated_gaussian_random_op.h"
#include "paddle/phi/core/generator.h"

namespace paddle {
namespace operators {

template <typename T>
class TruncatedGaussianRandomMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.Attr<float>("mean");
    float std = context.Attr<float>("std");
    auto* tensor = context.Output<phi::DenseTensor>("Out");
    tensor->mutable_data<T>(context.GetPlace());

    phi::DenseTensor cpu_tensor(tensor->dtype());
    cpu_tensor.Resize(tensor->dims());
    T* data_cpu = cpu_tensor.mutable_data<T>(platform::CPUPlace());

    std::uniform_real_distribution<T> dist(std::numeric_limits<float>::min(),
                                           1.0);
    TruncatedNormal<T> truncated_normal(mean, std);
    int64_t size = tensor->numel();

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    auto engine = phi::GetCPURandomEngine(seed);

    for (int64_t i = 0; i < size; ++i) {
      data_cpu[i] = truncated_normal(dist(*engine));
    }

    auto& dev_ctx =
        context.template device_context<platform::MLUDeviceContext>();
    framework::TensorCopy(cpu_tensor, context.GetPlace(), dev_ctx, tensor);
    dev_ctx.Wait();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(truncated_gaussian_random,
                       ops::TruncatedGaussianRandomMLUKernel<float>);
