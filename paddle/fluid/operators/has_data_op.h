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
#include <math.h>
#include <type_traits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise_op_function.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class HasDataOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::LoDTensor>("X");
    auto* output = context.Output<framework::LoDTensor>("Out");
    size_t mem_size = input->memory_size();

    auto* output_data =
      output->mutable_data<bool>(platform::CPUPlace());
    if (mem_size > 0) {
      output_data[0] = true;
    } else {
      output_data[0] = false;
    }
  }
};

}  // namespace operators
}  // namespace paddle
