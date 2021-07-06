/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <random>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/fill_constant_op.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
class NPUGaussianRandomKernel : public framework::OpKernel<T> {
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
    std::unique_ptr<T[]> data_cpu(new T[size]);
    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    auto engine = framework::GetCPURandomEngine(seed);

    for (int64_t i = 0; i < size; ++i) {
      data_cpu[i] = dist(*engine);
    }
    // copy to NPU
    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    memory::Copy(BOOST_GET_CONST(platform::NPUPlace, context.GetPlace()), data,
                 platform::CPUPlace(), reinterpret_cast<void*>(data_cpu.get()),
                 size * sizeof(T), stream);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(gaussian_random, ops::NPUGaussianRandomKernel<float>);
