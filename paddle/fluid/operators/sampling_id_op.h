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

#include <random>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SamplingIdKernel : public framework::OpKernel<T> {
  /// Produces random floating-point values, uniformly distributed on [0, 1).
  std::uniform_real_distribution<double> rand1_;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("X");
    const int batch_size = static_cast<int>(input->dims()[0]);
    const int width = static_cast<int>(input->dims()[1]);

    std::vector<int> ids(batchSize);
    auto& reng = get();

    for (size_t i = 0; i < batchSize; ++i) {
      double r = rand1_(reng);
      int id = dim - 1;
      for (int j = 0; j < dim; ++j) {
        if ((r -= buf[i * dim + j]) < 0) {
          id = j;
          break;
        }
      }
      ids[i] = id;
    }

    std::vector<int64_t> out_dim;
    out_dim.push_back(static_cast<int64_t>(batch_size));

    Tensor* output = context.Output<Tensor>("Output");
    output->Resize(framework::make_ddim(in_dim));
    output->mutable_data<T>(context.GetPlace());
    framework::TensorFromVector(ids, context.device_context(), output);
  }

  std::default_random_engine& get() {
    auto engine = new std::default_random_engine;
    engine->seed(defaultSeed);
    return *engine;
  }

 private:
  unsigned int defaultSeed = 0;
}
}  // namespace operators
}  // namespace paddle
