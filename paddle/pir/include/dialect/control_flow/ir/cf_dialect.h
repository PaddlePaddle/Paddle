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

#include "paddle/pir/include/core/dialect.h"

namespace pir {
class ControlFlowDialect : public Dialect {
 public:
  explicit ControlFlowDialect(IrContext *context)
      : Dialect(name(), context, TypeId::get<ControlFlowDialect>()) {
    initialize();
  }
  static const char *name() { return "cf"; }
  TEST_API void PrintType(Type type, std::ostream &os) const override;
  TEST_API OpPrintFn PrintOperation(Operation *op) const override;

 private:
  TEST_API void initialize();
};

}  // namespace pir
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ControlFlowDialect)
