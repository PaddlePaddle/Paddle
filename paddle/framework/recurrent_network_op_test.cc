/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include <gtest/gtest.h>

#include "paddle/framework/recurrent_network_op.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

class RecurrentOpTest : public ::testing::Test {
 protected:
  virtual void SetUp() override {
    CreateGlobalVariables();
    CreateRNNOp();
  }

  void CreateGlobalVariables() {
    // create boot memory
    scope.CreateVariable("h_boot");
    // create input, and init content
    Variable* x = scope.CreateVariable("x");
    DDim dims = make_ddim(std::vector<int>{10 /*sent size*/, 20 /*batch size*/,
                                           30 /*input dim*/});
    x->GetMutable<Tensor>()->mutable_data<float>(dims, platform::CPUPlace());
  }

  void CreateRNNOp() {
    OpDesc op_desc;

    op_desc.set_type("rnn_op");
    op_desc.add_inputs("x");
    // output hidden vectors
    op_desc.add_outputs("hiddens");

    auto memories_attr = op_desc.mutable_attrs()->Add();
    memories_attr->set_type(paddle::framework::AttrType::STRINGS);

    *memories_attr->mutable_strings()->Add() = "h";
    memories_attr->set_name("memories");

    auto boot_memories_attr = op_desc.mutable_attrs()->Add();
    boot_memories_attr->set_type(paddle::framework::AttrType::STRINGS);
    *boot_memories_attr->mutable_strings()->Add() = "h_boot";
    boot_memories_attr->set_name("boot_memories");

    AttributeMap attrs;
    attrs["memories"] = std::vector<std::string>{"h"};
    attrs["boot_memories"] = std::vector<std::string>{"h_boot"};

    rnn_op.Init(op_desc, attrs);
  }

  void RunRnnOp() {
    // TODO
  }

  // father scope
  Scope scope;
  RecurrentOp rnn_op;
};

TEST_F(RecurrentOpTest, create_op) {}

}  // namespace framework
}  // namespace paddle
