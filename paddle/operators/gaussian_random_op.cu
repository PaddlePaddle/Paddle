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

#include <memory>
#include <random>
#include "paddle/platform/dynload/curand.h"
#include "paddle/platform/gpu_info.h"

#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class GaussianRandomKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    float mean = context.op_.GetAttr<float>("mean");
    float std = context.op_.GetAttr<float>("std");
    auto* tensor = context.Output<framework::Tensor>(0);
    T* data = tensor->mutable_data<T>(context.GetPlace());

    int seed = context.op_.GetAttr<int>("seed");
    if (seed == 0) {
      seed = std::random_device()();
    }
    curandGenerator_t g;
    PADDLE_ENFORCE(platform::dynload::curandCreateGenerator(
        &g, CURAND_RNG_PSEUDO_DEFAULT));
    PADDLE_ENFORCE(
        platform::dynload::curandSetPseudoRandomGeneratorSeed(g, seed));
    platform::dynload::curandGenerateNormal(
        g, data, framework::product(tensor->dims()), mean, std);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(gaussian_random, ops::GaussianRandomKernel<float>);
