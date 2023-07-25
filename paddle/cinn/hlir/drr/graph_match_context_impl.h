// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <string>

#include "cinn/hlir/drr/match_context_impl.h"

namespace cinn {
namespace hlir {
namespace drr {

class TensorInterface;

class GraphMatchContextImpl : public MatchContextImpl {
 public:
  ~GraphMatchContextImpl() = default;

  const TensorInterface& Tensor(const std::string& tensor_name) const override;

 protected:
  std::type_index TypeIndex4Node() const override;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
