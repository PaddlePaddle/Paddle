/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

struct FillOpVisitor {
  FillOpVisitor(phi::DenseTensor *tensor, const std::vector<float> &value)
      : tensor_(tensor), value_(value) {}

  template <typename T>
  void apply() const {
    platform::CPUPlace cpu;
    auto *data = tensor_->mutable_data<T>(cpu);
    std::transform(
        value_.data(), value_.data() + tensor_->numel(), data, [](float dat) {
          return static_cast<T>(dat);
        });
  }

  phi::DenseTensor *tensor_;
  const std::vector<float> &value_;
};

}  // namespace operators
}  // namespace paddle
