/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SamplingIdKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("X");
    const int batch_size = static_cast<int>(input->dims()[0]);
    const int width = static_cast<int>(input->dims()[1]);

    PADDLE_ENFORCE_GE(
        batch_size, 0,
        platform::errors::InvalidArgument(
            "batch_size(dims[0]) must be nonnegative. but it is %d.",
            batch_size));
    PADDLE_ENFORCE_GE(
        width, 0,
        platform::errors::InvalidArgument(
            "width(dims[1]) must be nonnegative. but it is %d.", width));

    std::vector<T> ins_vector;
    framework::TensorToVector(*input, context.device_context(), &ins_vector);

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));

    std::uniform_real_distribution<T> dist(
        static_cast<T>(context.Attr<float>("min")),
        static_cast<T>(context.Attr<float>("max")));

    auto engine = framework::GetCPURandomEngine(seed);
    std::vector<int64_t> ids(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      T r = dist(*engine);
      int idx = width - 1;
      for (int j = 0; j < width; ++j) {
        if ((r -= ins_vector[i * width + j]) < 0) {
          idx = j;
          break;
        }
      }
      ids[i] = int64_t(idx);
    }

    std::vector<int64_t> out_dim;
    out_dim.push_back(static_cast<int64_t>(batch_size));

    Tensor* output = context.Output<Tensor>("Out");
    output->Resize(phi::make_ddim(out_dim));
    output->mutable_data<T>(context.GetPlace());
    framework::TensorFromVector(ids, context.device_context(), output);
  }
};

}  // namespace operators
}  // namespace paddle
