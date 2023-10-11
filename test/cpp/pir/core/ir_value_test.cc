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

#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/operation.h"

// This unittest is used to test the construction interfaces of value class and
// operation. The constructed test scenario is: a = OP1(); b = OP2(); c = OP3(a,
// b); d, e, f, g, h, i, j = OP4(a, c);
pir::AttributeMap CreateAttributeMap(std::string attribute_name,
                                     std::string attribute) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  pir::Attribute attr_value = pir::StrAttribute::get(ctx, attribute);
  pir::AttributeMap attr_map;
  attr_map.insert(
      std::pair<std::string, pir::Attribute>(attribute_name, attr_value));
  return attr_map;
}

TEST(value_test, value_test) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  // 1. Construct OP1: a = OP1()
  std::vector<pir::Value> op1_inputs = {};
  std::vector<pir::Type> op1_output_types = {pir::Float32Type::get(ctx)};
  pir::Operation *op1 =
      pir::Operation::Create(op1_inputs,
                             CreateAttributeMap("op1_name", "op1_attr"),
                             op1_output_types,
                             pir::OpInfo());
  op1->Print(std::cout);
  pir::OpResult a = op1->result(0);
  EXPECT_TRUE(a.use_empty());
  // 2. Construct OP2: b = OP2();
  std::vector<pir::Value> op2_inputs = {};
  std::vector<pir::Type> op2_output_types = {pir::Float32Type::get(ctx)};
  pir::Operation *op2 =
      pir::Operation::Create(op2_inputs,
                             CreateAttributeMap("op2_name", "op2_attr"),
                             op2_output_types,
                             pir::OpInfo());
  op2->Print(std::cout);
  pir::OpResult b = op2->result(0);
  EXPECT_TRUE(b.use_empty());
  // 3. Construct OP3: c = OP3(a, b);
  std::vector<pir::Value> op3_inputs{a, b};
  std::vector<pir::Type> op3_output_types = {pir::Float32Type::get(ctx)};
  pir::Operation *op3 =
      pir::Operation::Create(op3_inputs,
                             CreateAttributeMap("op3_name", "op3_attr"),
                             op3_output_types,
                             pir::OpInfo());

  EXPECT_TRUE(op1->result(0).HasOneUse());
  EXPECT_TRUE(op2->result(0).HasOneUse());
  op3->Print(std::cout);
  pir::OpResult c = op3->result(0);
  // 4. Construct OP4: d, e, f, g, h, i, j = OP4(a, c);
  std::vector<pir::Value> op4_inputs = {a, c};
  std::vector<pir::Type> op4_output_types;
  for (size_t i = 0; i < 7; i++) {
    op4_output_types.push_back(pir::Float32Type::get(ctx));
  }
  pir::Operation *op4 =
      pir::Operation::Create(op4_inputs,
                             CreateAttributeMap("op4_name", "op4_attr"),
                             op4_output_types,
                             pir::OpInfo());
  op4->Print(std::cout);

  // Test 1:
  EXPECT_EQ(op1->result(0).owner(), op1);
  EXPECT_EQ(op2->result(0).owner(), op2);
  EXPECT_EQ(op3->result(0).owner(), op3);
  EXPECT_EQ(op4->result(6).owner(), op4);

  // Test 2: op1_first_output -> op4_first_input
  pir::OpResult op1_first_output = op1->result(0);
  pir::OpOperand op4_first_input = op4->operand(0);

  EXPECT_EQ(op1_first_output.first_use(), op4_first_input);
  pir::OpOperand op3_first_input = op3->operand(0);

  EXPECT_EQ(op4_first_input.next_use(), op3_first_input);
  EXPECT_EQ(op3_first_input.next_use(), nullptr);

  // Test 3: Value iterator
  using my_iterator = pir::Value::UseIterator;
  my_iterator iter = op1->result(0).use_begin();
  EXPECT_EQ(iter.owner(), op4);
  ++iter;
  EXPECT_EQ(iter.owner(), op3);

  // Test 4: Value Replace Use
  // a = OP1(); b = OP2(); c = OP3(a, b); d, e, f, g, h, i, j = OP4(a, c);
  //
  c.ReplaceUsesWithIf(b, [](pir::OpOperand) { return true; });
  EXPECT_EQ(op4->operand_source(1), b);
  EXPECT_TRUE(c.use_empty());

  b.ReplaceAllUsesWith(a);
  EXPECT_EQ(op4->operand_source(1), a);
  EXPECT_TRUE(b.use_empty());

  // destroy
  VLOG(0) << op1->result(0).PrintUdChain() << std::endl;
  op4->Destroy();
  VLOG(0) << op1->result(0).PrintUdChain() << std::endl;
  op3->Destroy();
  VLOG(0) << op1->result(0).PrintUdChain() << std::endl;
  op2->Destroy();
  VLOG(0) << op1->result(0).PrintUdChain() << std::endl;
  op1->Destroy();
}
