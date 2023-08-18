// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>

#include "paddle/fluid/ir/drr/api/tensor_interface.h"
#include "paddle/fluid/ir/drr/ir_operation.h"

namespace ir {
namespace drr {

class TensorInterface;
class MatchContextImpl;

class MatchContext final {
 public:
  MatchContext(std::shared_ptr<const MatchContextImpl> impl);

  const TensorInterface& Tensor(const std::string& tensor_name) const;

  template <typename T>
  T Attr(const std::string& attr_name) const;

 private:
  std::shared_ptr<const MatchContextImpl> impl_;
};

}  // namespace drr
}  // namespace ir
