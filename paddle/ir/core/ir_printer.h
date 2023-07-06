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

#include <ostream>
#include <string>
#include <unordered_map>

#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/region.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value.h"

namespace ir {

class BasicIrPrinter {
 public:
  explicit BasicIrPrinter(std::ostream& os) : os(os) {}

  void PrintType(Type type) const;

  void PrintAttribute(Attribute attr) const;

 public:
  std::ostream& os;
};

class IR_API IrPrinter : public BasicIrPrinter {
 public:
  explicit IrPrinter(std::ostream& os) : BasicIrPrinter(os) {}

  /// @brief print program
  /// @param program
  void PrintProgram(const Program* program) const;

  /// @brief dispatch to custom printer function or PrintGeneralOperation
  void PrintOperation(const Operation* op) const;
  /// @brief print operation itself without its regions
  void PrintGeneralOperation(const Operation* op) const;
  /// @brief print operation and its regions
  void PrintFullOperation(const Operation* op) const;

  void PrintRegion(const Region& Region) const;
  void PrintBlock(const Block* block) const;

  void PrintValue(const Value& v) const;

  void PrintOpResult(const Operation* op) const;

  void PrintAttributeMap(const Operation* op) const;

  void PrintOpOperands(const Operation* op) const;

  void PrintOperandsType(const Operation* op) const;

  void PrintOpReturnType(const Operation* op) const;

 private:
  mutable size_t cur_var_number_{0};
  mutable std::unordered_map<const void*, std::string> aliases_;
};

}  // namespace ir
