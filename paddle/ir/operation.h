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

#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/type.h"
#include "paddle/ir/value_impl.h"

namespace ir {

class alignas(8) Operation final {
 public:
  ///
  /// \brief Malloc memory and construct objects in the following order:
  /// OpResultImpls|Operation|OpOperandImpls.
  ///
  static Operation *create(const std::vector<ir::OpResult> &inputs,
                           const std::vector<ir::Type> &output_types,
                           ir::DictionaryAttribute attribute);

  void destroy();

  ir::OpResult GetResultByIndex(uint32_t index);

  std::string print();

  ir::DictionaryAttribute attribute() { return attribute_; }

  uint32_t num_results() { return num_results_; }

  uint32_t num_operands() { return num_operands_; }

 private:
  Operation(uint32_t num_results,
            uint32_t num_operands,
            ir::DictionaryAttribute attribute);

  ir::DictionaryAttribute attribute_;

  uint32_t num_results_ = 0;

  uint32_t num_operands_ = 0;
};

}  // namespace ir
