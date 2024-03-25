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

#include "paddle/cinn/adt/adt.h"
#include "paddle/pir/include/core/value.h"

namespace cinn::adt::adapter {

struct Tensor final {
  ::pir::Value node_data;

  bool operator==(const Tensor& other) const {
    return this->node_data == other.node_data;
  }

  std::size_t GetRank() const;

  std::vector<int32_t> GetShape() const;

  std::size_t GetNumel() const;
};

inline std::size_t GetHashValueImpl(const Tensor& tensor) {
  return std::hash<::pir::Value>()(tensor.node_data);
}

}  // namespace cinn::adt::adapter
