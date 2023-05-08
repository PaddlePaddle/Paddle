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

// This unittest is used to test the construction interfaces of value class and
// operation. The constructed test scenario is: a = OP1(); b = OP2(); c = OP3(a,
// b); d, e, f, g, h, i, j = OP4(a, c);

ir::DictionaryAttribute CreateAttribute(std::string attribute_name,
                                        std::string attribute) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::StrAttribute attr_name = ir::StrAttribute::get(ctx, attribute_name);
  ir::Attribute attr_value = ir::StrAttribute::get(ctx, attribute);
  std::map<ir::StrAttribute, ir::Attribute> named_attr;
  named_attr.insert(
      std::pair<ir::StrAttribute, ir::Attribute>(attr_name, attr_value));
  return ir::DictionaryAttribute::get(ctx, named_attr);
}

TEST(value_test, value_test) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  // 1. Construct OP1: a = OP1()
  std::vector<ir::OpResult> op1_inputs = {};
  std::vector<ir::Type> op1_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op1 =
      ir::Operation::create(op1_inputs,
                            op1_output_types,
                            CreateAttribute("op1_name", "op1_attr"),
                            nullptr);
  std::cout << op1->print() << std::endl;
  // 2. Construct OP2: b = OP2();
  std::vector<ir::OpResult> op2_inputs = {};
  std::vector<ir::Type> op2_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op2 =
      ir::Operation::create(op2_inputs,
                            op2_output_types,
                            CreateAttribute("op2_name", "op2_attr"),
                            nullptr);
  std::cout << op2->print() << std::endl;
  // 3. Construct OP3: c = OP3(a, b);
  std::vector<ir::OpResult> op3_inputs = {op1->GetResultByIndex(0),
                                          op2->GetResultByIndex(0)};
  std::vector<ir::Type> op3_output_types = {ir::Float32Type::get(ctx)};
  ir::Operation *op3 =
      ir::Operation::create(op3_inputs,
                            op3_output_types,
                            CreateAttribute("op3_name", "op3_attr"),
                            nullptr);
  std::cout << op3->print() << std::endl;
  // 4. Construct OP4: d, e, f, g, h, i, j = OP4(a, c);
  std::vector<ir::OpResult> op4_inputs = {op1->GetResultByIndex(0),
                                          op3->GetResultByIndex(0)};
  std::vector<ir::Type> op4_output_types;
  for (size_t i = 0; i < 7; i++) {
    op4_output_types.push_back(ir::Float32Type::get(ctx));
  }
  ir::Operation *op4 =
      ir::Operation::create(op4_inputs,
                            op4_output_types,
                            CreateAttribute("op4_name", "op4_attr"),
                            nullptr);
  std::cout << op4->print() << std::endl;

  // Test 1:
  EXPECT_EQ(op1->GetResultByIndex(0).GetDefiningOp(), op1);
  EXPECT_EQ(op2->GetResultByIndex(0).GetDefiningOp(), op2);
  EXPECT_EQ(op3->GetResultByIndex(0).GetDefiningOp(), op3);
  EXPECT_EQ(op4->GetResultByIndex(6).GetDefiningOp(), op4);

  // Test 2: op1_first_output -> op4_first_input
  ir::OpResult op1_first_output = op1->GetResultByIndex(0);
  ir::detail::OpOperandImpl *op4_first_input =
      reinterpret_cast<ir::detail::OpOperandImpl *>(
          reinterpret_cast<uintptr_t>(op4) + sizeof(ir::Operation));
  EXPECT_EQ(static_cast<ir::Value>(op1_first_output).impl()->first_use(),
            op4_first_input);
  ir::detail::OpOperandImpl *op3_first_input =
      reinterpret_cast<ir::detail::OpOperandImpl *>(
          reinterpret_cast<uintptr_t>(op3) + sizeof(ir::Operation));
  EXPECT_EQ(op4_first_input->next_use(), op3_first_input);
  EXPECT_EQ(op3_first_input->next_use(), nullptr);

  // Test 3: Value iterator
  ir::Value::use_iterator iter = op1->GetResultByIndex(0).begin();
  EXPECT_EQ(iter.owner(), op4);
  ++iter;
  EXPECT_EQ(iter.owner(), op3);

  // destroy
  std::cout << op1->GetResultByIndex(0).print_ud_chain() << std::endl;
  op4->destroy();
  std::cout << op1->GetResultByIndex(0).print_ud_chain() << std::endl;
  op3->destroy();
  std::cout << op1->GetResultByIndex(0).print_ud_chain() << std::endl;
  op2->destroy();
  std::cout << op1->GetResultByIndex(0).print_ud_chain() << std::endl;
  op1->destroy();
}
