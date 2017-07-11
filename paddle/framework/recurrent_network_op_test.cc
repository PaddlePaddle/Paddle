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

#include <glog/logging.h>
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

  virtual void TearDown() {}

  void CreateGlobalVariables() {
    // create boot memory
    scope_.CreateVariable("h_boot");
    // create input, and init content
    // Variable* x = scope_.CreateVariable("x");
    DDim dims = make_ddim(std::vector<int>{10 /*sent size*/, 20 /*batch size*/,
                                           30 /*input dim*/});
    // TODO mutable_data is not valid
    // x->GetMutable<Tensor>()->mutable_data<float>(dims, platform::CPUPlace());
  }

  void CreateRNNOp() {
    OpDesc op_desc;

    op_desc.set_type("rnn_op");
    op_desc.add_inputs("x");
    // output hidden vectors
    op_desc.add_outputs("hiddens");

    // add memories
    auto memories_attr = op_desc.mutable_attrs()->Add();
    memories_attr->set_type(paddle::framework::AttrType::STRINGS);
    *memories_attr->mutable_strings()->Add() = "h";
    memories_attr->set_name("memories");

    // add initial memories
    auto boot_memories_attr = op_desc.mutable_attrs()->Add();
    boot_memories_attr->set_type(paddle::framework::AttrType::STRINGS);
    *boot_memories_attr->mutable_strings()->Add() = "h_boot";
    boot_memories_attr->set_name("boot_memories");

    // add step net desc
    auto step_net_attr = op_desc.mutable_attrs()->Add();
    step_net_attr->set_type(paddle::framework::AttrType::STRING);
    step_net_attr->set_s(" ");  // TODO add step net proto
    step_net_attr->set_name("step_net");

    std::ostringstream stream;
    op_desc.SerializeToOstream(&stream);
    std::string text = stream.str();
    LOG(INFO) << text;

    AttributeMap attrs;
    attrs["memories"] = std::vector<std::string>{"h"};
    attrs["boot_memories"] = std::vector<std::string>{"h_boot"};

    rnn_op_.Init(op_desc, attrs);
  }

  void RunRnnOp() {
    // TODO
  }

  // father scope
  Scope scope_;
  RecurrentOp rnn_op_;
};

TEST_F(RecurrentOpTest, create_op) {}

}  // namespace framework
}  // namespace paddle
