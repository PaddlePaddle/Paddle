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

#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/region.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/value.h"

namespace pir {

class BasicIrPrinter {
 public:
  explicit BasicIrPrinter(std::ostream& os) : os(os) {}

  void PrintType(Type type);

  void PrintAttribute(Attribute attr);

 public:
  std::ostream& os;
};

class IR_API IrPrinter : public BasicIrPrinter {
 public:
  explicit IrPrinter(std::ostream& os) : BasicIrPrinter(os) {}

  /// @brief print program
  /// @param program
  void PrintProgram(const Program* program);

  /// @brief dispatch to custom printer function or PrintGeneralOperation
  void PrintOperation(Operation* op);
  /// @brief print operation itself without its regions
  void PrintGeneralOperation(Operation* op);
  /// @brief print operation and its regions
  void PrintFullOperation(Operation* op);

  void PrintRegion(const Region& Region);
  void PrintBlock(const Block* block);

  void PrintValue(Value v);

  void PrintOpResult(Operation* op);

  void PrintAttributeMap(Operation* op);

  void PrintOpOperands(Operation* op);

  void PrintOperandsType(Operation* op);

  void PrintOpReturnType(Operation* op);

 private:
  size_t cur_var_number_{0};
  std::unordered_map<const void*, std::string> aliases_;
};

}  // namespace pir
