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

#include "paddle/framework/executor.h"

#include <memory>
#include <vector>

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/framework/attribute.h"
#include "paddle/framework/block_desc.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/program_desc.h"

USE_OP(elementwise_add);
USE_OP(gaussian_random);
USE_NO_KERNEL_OP(feed);
USE_NO_KERNEL_OP(fetch);
USE_OP(mul);
USE_OP(sum);
USE_OP(squared_l2_distance);
USE_OP(fill_constant);
USE_OP(mean);
USE_OP(sgd);
USE_OP(relu);
USE_NO_KERNEL_OP(recurrent);

constexpr auto kFeedValueName = "feed_value";
constexpr auto kFetchValueName = "fetch_value";

// using ProgramDescBind = paddle::framework::ProgramDescBind;
// using BlockDescBind = paddle::framework::BlockDescBind;
// using OpDescBind = paddle::framework::OpDescBind;
using namespace paddle::platform;
using namespace paddle::framework;

OpDescBind* AddOp(const std::string& type, const VariableNameMap& inputs,
                  const VariableNameMap& outputs, AttributeMap attrs,
                  BlockDescBind* block) {
  auto op = block->AppendOp();
  op->SetType(type);
  for (auto& kv : inputs) {
    op->SetInput(kv.first, kv.second);
  }
  for (auto& kv : outputs) {
    op->SetOutput(kv.first, kv.second);
  }
  op->SetAttrMap(attrs);
  op->CheckAttrs();

  return op;
}

class ExecutorTesterRandom : public ::testing::Test {
 public:
  virtual void SetUp() override {
    int seq_len = 2, input_dim = 3, batch_size = 1, embed_dim = 5;

    auto temp_init_root_block = init_pdesc_.add_blocks();
    temp_init_root_block->set_idx(0);
    temp_init_root_block->set_parent_idx(-1);
    ProgramDescBind& init_program = ProgramDescBind::Instance(&init_pdesc_);
    BlockDescBind* init_root_block = init_program.Block(0);

    AddOp("gaussian_random", {}, {{"Out", {"w1"}}},
          {{"dims", std::vector<int>{input_dim, embed_dim}}}, init_root_block);
    AddOp("gaussian_random", {}, {{"Out", {"w2"}}},
          {{"dims", std::vector<int>{embed_dim, input_dim}}}, init_root_block);
    init_root_block->Var("w1");
    init_root_block->Var("w2");

    // flush
    init_program.Proto();
    for (auto& var : *init_pdesc_.mutable_blocks(0)->mutable_vars()) {
      var.set_persistable(true);
    }

    // run block
    auto temp_root_block = pdesc_.add_blocks();
    temp_root_block->set_idx(0);
    temp_root_block->set_parent_idx(-1);
    ProgramDescBind& program = ProgramDescBind::Instance(&pdesc_);
    BlockDescBind* root_block = program.Block(0);

    // feed data
    AddOp("gaussian_random", {}, {{"Out", {"a"}}},
          {{"dims", std::vector<int>{seq_len, batch_size, input_dim}}},
          root_block);
    root_block->Var("a");
    AddOp("gaussian_random", {}, {{"Out", {"h_boot"}}},
          {{"dims", std::vector<int>{batch_size, input_dim}}}, root_block);
    root_block->Var("h_boot");

    root_block->Var("b");
    root_block->Var("step_scopes");
    auto rnn_op =
        AddOp("recurrent", {{"inlinks", {"a"}}, {"boot_memories", {"h_boot"}}},
              {{"outlinks", {"b"}}, {"step_scopes", {"step_scopes"}}},
              {{"pre_memories", std::vector<std::string>{"h@pre"}},
               {"memories", std::vector<std::string>{"h@mem"}}},
              root_block);

    BlockDescBind* second_block = program.AppendBlock(*root_block);
    rnn_op->SetBlockAttr("block_idx", *second_block);
    AddOp("elementwise_add", {{"X", {"a"}}, {"Y", {"h@pre"}}},
          {{"Out", {"h@mem"}}}, {}, second_block);
    AddOp("relu", {{"X", {"h@mem"}}}, {{"Y", {"b"}}}, {}, second_block);
    second_block->Var("a");
    second_block->Var("b");
    second_block->Var("h@pre");
    second_block->Var("h@mem");

    // flush
    program.Proto();
  }

 protected:
  ProgramDesc init_pdesc_;
  ProgramDesc pdesc_;
  std::vector<std::vector<float>> inputs_;
};

TEST_F(ExecutorTesterRandom, CPU) {
  std::vector<Place> places;
  CPUPlace cpu_place;
  places.push_back(cpu_place);

  // We have a global Scope and BuddyAllocator, and we must ensure
  // global BuddyAllocator is initialized before global Scope. Thus,
  // global Scope will deconstruct before BuddyAllocator. Otherwise,
  // "pointer being freed was not allocated" error will appear.
  paddle::memory::Used(cpu_place);

  std::unique_ptr<Executor> executor(new Executor(places));
  executor->Run(init_pdesc_, &GetGlobalScope(), 0);
  executor->Run(pdesc_, &GetGlobalScope(), 0);
}
