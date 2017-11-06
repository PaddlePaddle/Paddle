/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/prune.h"

#include "paddle/framework/attribute.h"
#include "paddle/framework/operator.h"
#include "paddle/operators/net_op.h"

#include "paddle/framework/block_desc.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/program_desc.h"

#include <gtest/gtest.h>

namespace f = paddle::framework;
namespace ops = paddle::operators;

void AddOp(const std::string &type, const f::VariableNameMap &inputs,
           const f::VariableNameMap &outputs, f::AttributeMap attrs,
           paddle::framework::BlockDescBind *block) {
  // insert output
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      auto var = block->Var(v);
      var->SetDataType(paddle::framework::DataType::FP32);
    }
  }

  // insert op
  auto op = block->AppendOp();
  op->SetType(type);
  for (auto &kv : inputs) {
    op->SetInput(kv.first, kv.second);
  }
  for (auto &kv : outputs) {
    op->SetOutput(kv.first, kv.second);
  }
  op->SetAttrMap(attrs);
}

TEST(Prune, one_operator) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);

  AddOp("one_one", {{"input", {"a"}}}, {{"output", {"b"}}}, {}, block);

  f::ProgramDesc *pdesc = program.Proto();
  f::ProgramDesc pruned;

  Prune(*pdesc, &pruned);
  PADDLE_ENFORCE_EQ(pruned.blocks(0).ops_size(), 0);

  pdesc->mutable_blocks(0)->mutable_ops(0)->set_is_target(true);
  Prune(*pdesc, &pruned);
  PADDLE_ENFORCE_EQ(pruned.blocks(0).ops_size(), 1);
}

TEST(Prune, forward) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);

  AddOp("one_one", {{"input", {"a"}}}, {{"output", {"b"}}}, {}, block);
  AddOp("one_one", {{"input", {"b"}}}, {{"output", {"c"}}}, {}, block);
  AddOp("one_one", {{"input", {"c"}}}, {{"output", {"d"}}}, {}, block);
  AddOp("one_one", {{"input", {"d"}}}, {{"output", {"e"}}}, {}, block);

  f::ProgramDesc *pdesc = program.Proto();

  for (int i = 0; i < pdesc->blocks(0).ops_size(); ++i) {
    f::ProgramDesc pruned;
    pdesc->mutable_blocks(0)->mutable_ops(i)->set_is_target(true);
    Prune(*pdesc, &pruned);
    PADDLE_ENFORCE_EQ(pruned.blocks(0).ops_size(), i + 1);
  }
}

TEST(Prune, multi_input_op) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);

  AddOp("one_one", {{"input", {"a0"}}}, {{"output", {"b0"}}}, {}, block);
  AddOp("one_one", {{"input", {"a1"}}}, {{"output", {"b1"}}}, {}, block);
  AddOp("one_one", {{"input", {"a2"}}}, {{"output", {"b2"}}}, {}, block);
  AddOp("three_one", {{"input", {"b0", "b1", "b2"}}}, {{"output", {"c"}}}, {},
        block);

  f::ProgramDesc *pdesc = program.Proto();
  pdesc->mutable_blocks(0)->mutable_ops(3)->set_is_target(true);

  f::ProgramDesc pruned;
  Prune(*pdesc, &pruned);
  PADDLE_ENFORCE_EQ(pruned.blocks(0).ops_size(), 4);
}

TEST(Prune, multi_output_op) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);

  AddOp("one_two", {{"input", {"a"}}}, {{"output", {"b", "c"}}}, {}, block);
  AddOp("one_one", {{"input", {"b"}}}, {{"output", {"b1"}}}, {}, block);
  AddOp("one_one", {{"input", {"c"}}}, {{"output", {"c1"}}}, {}, block);

  f::ProgramDesc *pdesc = program.Proto();
  pdesc->mutable_blocks(0)->mutable_ops(2)->set_is_target(true);

  f::ProgramDesc pruned;
  Prune(*pdesc, &pruned);
  PADDLE_ENFORCE_EQ(pruned.blocks(0).ops_size(), 2);
}

TEST(Prune, multi_target) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.MutableBlock(0);

  AddOp("one_two", {{"input", {"a"}}}, {{"output", {"b", "c"}}}, {}, block);
  AddOp("one_one", {{"input", {"b"}}}, {{"output", {"b1"}}}, {}, block);
  AddOp("one_one", {{"input", {"c"}}}, {{"output", {"c1"}}}, {}, block);

  f::ProgramDesc *pdesc = program.Proto();
  pdesc->mutable_blocks(0)->mutable_ops(1)->set_is_target(true);
  pdesc->mutable_blocks(0)->mutable_ops(2)->set_is_target(true);

  f::ProgramDesc pruned;
  Prune(*pdesc, &pruned);
  PADDLE_ENFORCE_EQ(pruned.blocks(0).ops_size(), 3);
}
