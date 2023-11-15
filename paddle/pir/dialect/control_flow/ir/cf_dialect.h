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

#include "paddle/pir/core/dialect.h"

namespace pir {
class ControlFlowDialect : public Dialect {
 public:
  explicit ControlFlowDialect(IrContext *context)
      : Dialect(name(), context, TypeId::get<ControlFlowDialect>()) {
    initialize();
  }
  static constexpr char *name() { return "cf"; }
  void PrintType(pir::Type type, std::ostream &os) const override;
  void PrintOperation(pir::Operation *op,
                      pir::IrPrinter &printer) const override;  // NOLINT
 private:
  void initialize();
};

}  // namespace pir
IR_DECLARE_EXPLICIT_TYPE_ID(pir::ControlFlowDialect)
