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
#include <unordered_set>

#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/region.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/value.h"

namespace pir {

struct PrinterOptions {
  PrinterOptions() = default;
  PrinterOptions(const PrinterOptions&) = default;

  bool print_regions = true;

  // `custom_attrs` means attributes not in op defintion but added by users
  // by default, we only print attributes in definition
  std::unordered_set<std::string> custom_attrs_white_list = {
      kAttrStopGradients,
      kAttrIsPersisable,
  };
};

class BasicIrPrinter {
 public:
  explicit BasicIrPrinter(std::ostream& os,
                          const PrinterOptions& options = PrinterOptions())
      : os_(os), options_(options) {}

  void PrintType(Type type);

  void PrintAttribute(Attribute attr);

  std::ostream& stream() { return os_; }

 protected:
  std::ostream& os_;
  PrinterOptions options_;
};

class IR_API IrPrinter : public BasicIrPrinter {
 public:
  explicit IrPrinter(std::ostream& os,
                     const PrinterOptions& options = PrinterOptions())
      : BasicIrPrinter(os, options) {}

  /// @brief print program
  /// @param program
  void PrintProgram(const Program* program);

  /// @brief dispatch to custom printer function or PrintGeneralOperation
  void PrintOperation(Operation* op);
  /// @brief print operation itself without its regions
  void PrintOperationWithNoRegion(Operation* op);
  /// @brief print operation and its regions
  void PrintGeneralOperation(Operation* op);

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
