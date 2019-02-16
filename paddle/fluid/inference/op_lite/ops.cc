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

#include "paddle/fluid/inference/op_lite/ops.h"
#include <string>
#include "paddle/fluid/inference/op_lite/activation_op.h"
#include "paddle/fluid/inference/op_lite/fc_op.h"

namespace paddle {
namespace inference {
namespace op_lite {

LiteOpRegistry::LiteOpRegistry() {
  creators_.emplace("fc", []() -> std::unique_ptr<OpLite> {
    return std::unique_ptr<OpLite>(new FC);
  });
  creators_.emplace("relu", []() -> std::unique_ptr<OpLite> {
    return std::unique_ptr<OpLite>(new ReLU);
  });
}

std::unique_ptr<OpLite> LiteOpRegistry::Create(const std::string &op_type) {
  auto it = creators_.find(op_type);
  PADDLE_ENFORCE(it != creators_.end(), "No lite op creator called %s",
                 op_type);
  return it->second();
}

bool LiteOpRegistry::Has(const std::string &op_type) {
  return creators_.count(op_type);
}

}  // namespace op_lite
}  // namespace inference
}  // namespace paddle
