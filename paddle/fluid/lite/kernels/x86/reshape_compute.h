// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/operators/reshape_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class ReshapeCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ReshapeParam;

  void Run() override {
    auto &param = *param_.get_mutable<operators::ReshapeParam>();
    // auto& context = context_->As<X86Context>();
    CHECK(param.output);
    CHECK(param.x);

    auto *shape_tensor = param.actual_shape;
    lite::DDim out_dims = param.output->dims();
    if (shape_tensor) {
      auto *shape_data = shape_tensor->data<int>();
      auto shape = std::vector<int>(
          shape_data, shape_data + shape_tensor->dims().production());
      out_dims = paddle::lite::operators::ValidateShape(shape, param.x->dims());
    }

    param.output->mutable_data<T>();
    framework::TensorCopy(param.x->raw_tensor(), platform::CPUPlace(),
                          platform::CPUDeviceContext(),
                          &param.output->raw_tensor());
    param.output->Resize(out_dims);
  }
  virtual ~ReshapeCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
