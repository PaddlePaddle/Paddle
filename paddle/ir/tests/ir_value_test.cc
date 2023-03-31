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

#include <gtest/gtest.h>

#include "paddle/ir/attribute.h"
#include "paddle/ir/builtin_attribute.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/ir_context.h"
#include "paddle/ir/operation.h"

// Test case is:
// a = OP1();
// b = OP2();
// c = OP3(a, b);
// d, e, f, g, h, i, j = OP4(a, c);
// e = OP5(b, c, d, e, f, g, h, i, j);

ir::DictionaryAttribute GetOpAttribute(std::string attribute_name,
                                       std::string attribute) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::StrAttribute attr_name = ir::StrAttribute::get(ctx, attribute_name);
  ir::Attribute attr_value = ir::StrAttribute::get(ctx, attribute);
  std::map<ir::StrAttribute, ir::Attribute> named_attr;
  named_attr.insert(
      std::pair<ir::StrAttribute, ir::Attribute>(attr_name, attr_value));
  return ir::DictionaryAttribute::get(ctx, named_attr);
}

void print_ud_chain(ir::Operation *op, uint32_t idx) {
  std::cout << op->GetResultByIndex(idx).value_impl() << "->";
  ir::detail::OpOperandImpl *a =
      op->GetResultByIndex(idx).value_impl()->first_user();
  if (a) {
    std::cout << reinterpret_cast<void *>(a) << "->";
    while (a->next_user() != nullptr) {
      std::cout << reinterpret_cast<void *>(a)->next_user() << "->";
      a = a->next_user();
    }
  }
  std::cout << "nullptr" << std::endl;
}

TEST(value_test, value_test) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  // a = OP1();
  std::vector<ir::OpResult> op1_inputs = {};
  std::vector<ir::Type> op1_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op1 = ir::Operation::create(
      op1_inputs, op1_output_types, GetOpAttribute("op1_name", "op1_attr"));
  std::cout << "op1 ptr is: " << op1 << std::endl;
  // b = OP2();
  std::vector<ir::OpResult> op2_inputs = {};
  std::vector<ir::Type> op2_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op2 = ir::Operation::create(
      op2_inputs, op2_output_types, GetOpAttribute("op2_name", "op2_attr"));
  std::cout << "op2 ptr is: " << op2 << std::endl;
  // c = OP3(a, b);
  std::vector<ir::OpResult> op3_inputs = {op1->GetResultByIndex(0),
                                          op2->GetResultByIndex(0)};
  std::vector<ir::Type> op3_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op3 = ir::Operation::create(
      op3_inputs, op3_output_types, GetOpAttribute("op3_name", "op3_attr"));
  std::cout << "op3 ptr is: " << op3 << std::endl;
  // d, e, f, g, h, i, j = OP4(a, c);
  std::vector<ir::OpResult> op4_inputs = {op1->GetResultByIndex(0),
                                          op3->GetResultByIndex(0)};
  std::vector<ir::Type> op4_output_types;
  for (size_t i = 0; i < 7; i++) {
    op4_output_types.push_back(ir::Float32Type::get(ctx));
  }
  ir::Operation *op4 = ir::Operation::create(
      op4_inputs, op4_output_types, GetOpAttribute("op4_name", "op4_attr"));
  std::cout << "op4 ptr is: " << op4 << std::endl;
  // k = OP5(b, c, d, e, f, g, h, i, j);
  std::vector<ir::OpResult> op5_inputs = {op2->GetResultByIndex(0),
                                          op3->GetResultByIndex(0),
                                          op4->GetResultByIndex(6),
                                          op4->GetResultByIndex(5),
                                          op4->GetResultByIndex(4),
                                          op4->GetResultByIndex(3),
                                          op4->GetResultByIndex(2),
                                          op4->GetResultByIndex(1),
                                          op4->GetResultByIndex(0)};
  std::vector<ir::Type> op5_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op5 = ir::Operation::create(
      op5_inputs, op5_output_types, GetOpAttribute("op5_name", "op5_attr"));
  std::cout << "op5 ptr is: " << op5 << std::endl;
  // print
  std::cout << op1->print() << std::endl;
  std::cout << op2->print() << std::endl;
  std::cout << op3->print() << std::endl;
  std::cout << op4->print() << std::endl;
  std::cout << op5->print() << std::endl;
  // ud-chain
  std::cout << "op1 is: " << op1->GetResultByIndex(0).GetDefiningOp()
            << std::endl;
  std::cout << "op2 is: " << op2->GetResultByIndex(0).GetDefiningOp()
            << std::endl;
  std::cout << "op3 is: " << op3->GetResultByIndex(0).GetDefiningOp()
            << std::endl;
  std::cout << "op4 is: " << op4->GetResultByIndex(6).GetDefiningOp()
            << std::endl;
  std::cout << "op5 is: " << op5->GetResultByIndex(0).GetDefiningOp()
            << std::endl;

  print_ud_chain(op1, 0);
  print_ud_chain(op2, 0);
  print_ud_chain(op3, 0);
  print_ud_chain(op4, 0);
  print_ud_chain(op4, 1);
  print_ud_chain(op4, 2);
  print_ud_chain(op4, 3);
  print_ud_chain(op4, 4);
  print_ud_chain(op4, 5);
  print_ud_chain(op4, 6);
  print_ud_chain(op5, 0);

  // destroy
  op1->destroy();
  op2->destroy();
  op3->destroy();
  op4->destroy();
  op5->destroy();
}
