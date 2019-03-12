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

#include <boost/variant.hpp>
#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/op_lite/op_lite.h"

namespace paddle {
namespace inference {
namespace op_lite {
using framework::LoDTensor;

struct FcParam {
  LoDTensor* input{nullptr};
  LoDTensor* w{nullptr};
  LoDTensor* bias{nullptr};
  LoDTensor* output{nullptr};
  int in_num_col_dims{0};
};

class FC final : public OpLite {
 public:
  bool CheckShape() const override;
  bool InferShape() const override;
  bool Run() override;
  bool Build(const paddle::framework::OpDesc& opdesc,
             framework::Scope* scope) override;
  std::string DebugString() const override;

 private:
  FcParam param_;
};

}  // namespace op_lite
}  // namespace inference
}  // namespace paddle
