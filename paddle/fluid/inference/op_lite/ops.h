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

/**
 * In this file, all the lite operators are manually registered here.
 * We discard the factory registry way to get better control on the growth of
 * operators.
 */
#pragma once

#include <string>
#include "paddle/fluid/inference/op_lite/fc_op.h"

namespace paddle {
namespace inference {
namespace op_lite {

class LiteOpRegistry {
 public:
  using op_creator_t = std::function<std::unique_ptr<OpLite>()>;
  LiteOpRegistry();

  static LiteOpRegistry& Global() {
    static auto* x = new LiteOpRegistry;
    return *x;
  }

  std::unique_ptr<OpLite> Create(const std::string& op_type);
  bool Has(const std::string& op_type);

 private:
  std::unordered_map<std::string, op_creator_t> creators_;
};

}  // namespace op_lite
}  // namespace inference
}  // namespace paddle
