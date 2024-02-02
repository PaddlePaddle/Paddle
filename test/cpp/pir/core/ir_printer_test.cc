// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <sstream>

#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/program.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(printer_test, custom_hooks) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Dialect* test_dialect = ctx->GetOrRegisterDialect<test::TestDialect>();
  EXPECT_EQ(test_dialect != nullptr, true);

  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(test::Operation1::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(test::Operation2::name());

  pir::Operation* op1 = pir::Operation::Create(
      {},
      test::CreateAttributeMap({"op1_attr1", "op1_attr2"},
                               {"op1_attr1", "op1_attr2"}),
      {pir::Float32Type::get(ctx)},
      op1_info);
  pir::Operation* op2 = pir::Operation::Create(
      {op1->result(0)}, {}, {pir::Float32Type::get(ctx)}, op2_info);

  pir::Program program(ctx);
  program.block()->push_back(op1);
  program.block()->push_back(op2);

  pir::PrintHooks hooks;
  // this one retains old printing and adds new info
  hooks.value_print_hook = [](pir::Value v, pir::IrPrinter& printer) {
    printer.IrPrinter::PrintValue(v);
    printer.os << " [extra info]";
  };
  // this one overrides old printing
  hooks.op_print_hook = [](pir::Operation* op, pir::IrPrinter& printer) {
    printer.PrintOpResult(op);
    printer.os << " :=";

    printer.os << " \"" << op->name() << "\"";
    printer.PrintOpOperands(op);
    printer.PrintAttributeMap(op);
    printer.os << " :";
    printer.PrintOpReturnType(op);
    printer.os << "\n";
  };

  hooks.attribute_print_hook = [](pir::Attribute attr,
                                  pir::IrPrinter& printer) {
    printer.os << "[PlaceHolder]";
  };
  hooks.type_print_hook = [](pir::Type type, pir::IrPrinter& printer) {
    printer.os << "[" << type << "]";
  };

  std::stringstream ss;

  ss << pir::CustomPrintHelper{program, hooks};
  EXPECT_EQ(
      ss.str(),
      "{\n"
      "(%0 [extra info]) := \"test.operation1\" () "
      "{op1_attr1:[PlaceHolder],op1_attr2:[PlaceHolder]} :[f32]\n"
      "(%1 [extra info]) := \"test.operation2\" (%0 [extra info]) {} :[f32]\n"
      "}\n");
}
