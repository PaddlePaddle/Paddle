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

// fake op implementations
namespace fake {
class FcOp : public OperatorBase {
 public:
  FcOp(const OpDesc& desc) : name_(desc.name()) {}

  virtual void InferShape(ScopePtr scope) const override {
    for (const auto& output : outputs_) {
      LOG(INFO) << "fc [" << name_ << "]"
                << " create output variable [" << output << "]";
      scope->CreateVariable(output);
    }
  }

  virtual void Run(OpContext* contex) const override {
    for (const auto& input : inputs_) {
      PADDLE_ENFORCE(contex->scope->HasVariable(input),
                     "no input variable [%s] exists");
      LOG(INFO) << "fc [" << name_ << "] read input [" << input << "]";
    }
    for (const auto& output : outputs_) {
      PADDLE_ENFORCE(contex->scope->HasVariable(output),
                     "no output variable [%s] exists");
      LOG(INFO) << "fc [" << name_ << "] write output [" << output << "]";
    }
  }

 private:
  std::string name_;
};

class AddOp : public OperatorBase {
 public:
  AddOp(const OpDesc& desc) : name_(desc.name()) {}

  virtual void InferShape(ScopePtr scope) const override {
    for (const auto& output : outputs_) {
      LOG(INFO) << "add [" << name_ << "]"
                << " create output variable [" << output << "]";
      scope->CreateVariable(output);
    }
  }

  virtual void Run(OpContext* contex) const override {
    for (const auto& input : inputs_) {
      PADDLE_ENFORCE(contex->scope->HasVariable(input),
                     "no input variable [%s] exists");
      LOG(INFO) << "add [" << name_ << "] read input [" << input << "]";
    }
    for (const auto& output : outputs_) {
      PADDLE_ENFORCE(contex->scope->HasVariable(output),
                     "no output variable [%s] exists");
      LOG(INFO) << "add [" << name_ << "] write output [" << output << "]";
    }
  }

 private:
  std::string name_;
};
}  // namespace fake

void PlainNet::AddOp(const OpDesc& desc) {
  if (desc.type() == "fc") {
    ops_.emplace_back(new fake::FcOp(desc));
  } else if (desc.type() == "add") {
    ops_.emplace_back(new fake::AddOp(desc));
  }
}

class RecurrentOpTest : public ::testing::Test {
 protected:
  virtual void SetUp() override {
    CreateGlobalVariables();
    CreateStepNet();
    CreateRNNOp();
  }

  virtual void TearDown() override {}

  void CreateGlobalVariables() {
    LOG(INFO) << "create global variable h_boot";
    // create boot memory
    scope_.CreateVariable("h_boot");
    // create input, and init content
    LOG(INFO) << "create global variale x";
    Variable* x = scope_.CreateVariable("x");
    DDim dims = make_ddim(std::vector<int>{10 /*sent size*/, 20 /*batch size*/,
                                           30 /*input dim*/});
    // TODO mutable_data is not valid
    x->GetMutable<Tensor>()->mutable_data<float>(dims, platform::CPUPlace());
  }

  void CreateRNNOp() {
    OpDesc op_desc;

    op_desc.set_type("rnn_op");
    op_desc.set_name("simple_rnn");
    op_desc.add_inputs("x");
    op_desc.add_inputs("w");
    op_desc.add_inputs("h_boot");       // initial memory
    op_desc.add_inputs("step_net");     // step net
    op_desc.add_inputs("step_scopes");  // step scopes
    // output hidden vectors
    op_desc.add_outputs("h");

    // add memories
    auto memories_attr = op_desc.mutable_attrs()->Add();
    memories_attr->set_type(paddle::framework::AttrType::STRINGS);
    *memories_attr->mutable_strings()->Add() = "h";
    memories_attr->set_name("memories");

    // add memories
    auto pre_memories_attr = op_desc.mutable_attrs()->Add();
    pre_memories_attr->set_type(paddle::framework::AttrType::STRINGS);
    *pre_memories_attr->mutable_strings()->Add() = "h_pre";
    pre_memories_attr->set_name("pre_memories");

    // add initial memories
    auto boot_memories_attr = op_desc.mutable_attrs()->Add();
    boot_memories_attr->set_type(paddle::framework::AttrType::STRINGS);
    *boot_memories_attr->mutable_strings()->Add() = "h_boot";
    boot_memories_attr->set_name("boot_memories");

    // add step scopes
    auto step_scopes_attr = op_desc.mutable_attrs()->Add();
    step_scopes_attr->set_type(paddle::framework::AttrType::STRING);
    step_scopes_attr->set_s("step_scopes");
    step_scopes_attr->set_name("step_scopes");

    // add step net desc
    auto step_net_attr = op_desc.mutable_attrs()->Add();
    step_net_attr->set_type(paddle::framework::AttrType::STRING);
    step_net_attr->set_s("step_net");
    step_net_attr->set_name("step_net");

    // std::ostringstream stream;
    // op_desc.SerializeToOstream(&stream);
    // std::string text = stream.str();
    // LOG(INFO) << text;

    AttributeMap attrs;
    attrs["memories"] = std::vector<std::string>{"h"};
    attrs["pre_memories"] = std::vector<std::string>{"h_pre"};
    attrs["boot_memories"] = std::vector<std::string>{"h_boot"};
    attrs["step_net"] = std::vector<std::string>{"step_net"};
    attrs["step_scopes"] = std::vector<std::string>{"step_scopes"};

    // TODO
    LOG(INFO) << "rnn_op to init";
    rnn_op_.Init(op_desc, attrs);
    LOG(INFO) << "rnn_op finish init";
  }

  OpDesc CreateFcOpDesc() {
    OpDesc op_desc;
    op_desc.set_type("fc");
    op_desc.set_name("fc");
    op_desc.add_inputs("h_pre");
    op_desc.add_inputs("w");
    op_desc.add_outputs("s");
    return op_desc;
  }

  OpDesc CreateAddOpDesc() {
    OpDesc op_desc;
    op_desc.set_type("add");
    op_desc.set_name("add");
    op_desc.add_inputs("x");
    op_desc.add_inputs("s");
    op_desc.add_outputs("h");
    return op_desc;
  }

  void CreateStepNet() {
    Variable* net_var = scope_.CreateVariable("simple_rnn_net");
    NetDesc net_desc;
    net_desc.name_ = "simple_rnn_net";
    net_desc.op_descs.push_back(CreateFcOpDesc());
    net_desc.op_descs.push_back(CreateAddOpDesc());
    net_var->Reset<PlainNet>(new PlainNet(net_desc));
  }

  // father scope
  Scope scope_;
  RecurrentOp rnn_op_;
};

TEST_F(RecurrentOpTest, create_op) {}

TEST_F(RecurrentOpTest, Run) {
  OpContext ctx;
  ctx.scope = std::make_shared<Scope>();
  rnn_op_.Run(&ctx);
}

}  // namespace framework
}  // namespace paddle
