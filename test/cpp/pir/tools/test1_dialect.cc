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
#include "test/cpp/pir/tools/test1_dialect.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "test/cpp/pir/tools/test_op.h"

namespace test1 {

Test1Dialect::Test1Dialect(pir::IrContext *context)
    : pir::Dialect(name(), context, pir::TypeId::get<Test1Dialect>()) {
  initialize();
}
void Test1Dialect::initialize() {
  RegisterOps<Operation1, Operation2, Operation3, Operation4>();
}

pir::OpPrintFn Test1Dialect::PrintOperation(const pir::Operation &op) const {
  return [](const pir::Operation &op, pir::IrPrinter &printer) {
    printer.PrintOpResult(op);
    printer.os << " =";

    printer.os << " \"" << op.name() << "\"";
    printer.PrintOpOperands(op);
  };
}
}  // namespace test1
IR_DEFINE_EXPLICIT_TYPE_ID(test1::Test1Dialect)
