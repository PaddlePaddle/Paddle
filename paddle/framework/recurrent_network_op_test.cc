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
  FcOp(const OpDesc& desc) {}

  virtual void InferShape(ScopePtr scope) const override {
    for (const auto& output : outputs_) {
      LOG(INFO) << "fc [" << name_ << "]"
                << " create output variable [" << output << "]";
      scope->CreateVariable(output);
    }
  }

  virtual void Run(OpContext* contex) const override {
    LOG(INFO) << "run fc op";
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
  AddOp(const OpDesc& desc) {}

  virtual void InferShape(ScopePtr scope) const override {
    for (const auto& output : outputs_) {
      LOG(INFO) << "add [" << name_ << "]"
                << " create output variable [" << output << "]";
      scope->CreateVariable(output);
    }
  }

  virtual void Run(OpContext* contex) const override {
    LOG(INFO) << "run add op";
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
    scope_ = std::make_shared<Scope>();
    LOG(INFO) << "create global variable h_boot";
    // create boot memory
    scope_->CreateVariable("h_boot");
    // create input, and init content
    LOG(INFO) << "create global variable x";
    Variable* x = scope_->CreateVariable("x");
    DDim dims = make_ddim(std::vector<int>{10 /*sent size*/, 20 /*batch size*/,
                                           30 /*input dim*/});
    x->GetMutable<Tensor>()->mutable_data<float>(dims, platform::CPUPlace());

    LOG(INFO) << "create global variable w";
    Variable* w = scope_->CreateVariable("rnn/w");
    w->GetMutable<Tensor>()->mutable_data<float>(
        make_ddim(std::vector<int>{30, 30}), platform::CPUPlace());

    LOG(INFO) << "create global variable h_boot";
    Variable* h_boot = scope_->CreateVariable("h_boot");
    h_boot->GetMutable<Tensor>()->mutable_data<float>(
        make_ddim(std::vector<int>{20 /*batch size*/, 30 /*input dim*/}),
        platform::CPUPlace());

    LOG(INFO) << "create variable step_scopes";
    scope_->CreateVariable("step_scopes");

    LOG(INFO) << "create variable h";
    scope_->CreateVariable("h");
  }

  void CreateRNNOp() {
    OpDesc op_desc;

    op_desc.set_type("rnn_op");
    op_desc.add_inputs("x");
    op_desc.add_inputs("h_boot");    // initial memory
    op_desc.add_inputs("step_net");  // step net
    // output hidden vectors
    op_desc.add_outputs("h");
    op_desc.add_outputs("step_scopes");  // step scopes

    // add real input
    auto input_attr = op_desc.mutable_attrs()->Add();
    input_attr->set_type(paddle::framework::AttrType::INTS);
    *input_attr->mutable_ints()->Add() = 0;
    input_attr->set_name("in_links");

    // add input alias, this alias is used in step net.
    auto input_alias_attr = op_desc.mutable_attrs()->Add();
    input_alias_attr->set_type(paddle::framework::AttrType::STRINGS);
    *input_alias_attr->mutable_strings()->Add() = "rnn/x";
    input_alias_attr->set_name("in_link_alias");

    // add output alias, this alias is used in step net.
    auto output_alias_attr = op_desc.mutable_attrs()->Add();
    output_alias_attr->set_type(paddle::framework::AttrType::STRINGS);
    *output_alias_attr->mutable_strings()->Add() = "rnn/h";
    output_alias_attr->set_name("out_link_alias");

    // add memories
    auto memories_attr = op_desc.mutable_attrs()->Add();
    memories_attr->set_type(paddle::framework::AttrType::STRINGS);
    *memories_attr->mutable_strings()->Add() = "rnn/h";
    memories_attr->set_name("memories");

    // add history/previous memories
    auto pre_memories_attr = op_desc.mutable_attrs()->Add();
    pre_memories_attr->set_type(paddle::framework::AttrType::STRINGS);
    *pre_memories_attr->mutable_strings()->Add() = "rnn/h_pre";
    pre_memories_attr->set_name("pre_memories");

    // add initial memories
    auto boot_memories_attr = op_desc.mutable_attrs()->Add();
    boot_memories_attr->set_type(paddle::framework::AttrType::INTS);
    *boot_memories_attr->mutable_ints()->Add() = 1;
    boot_memories_attr->set_name("boot_memories");

    // add step net desc
    auto step_net_attr = op_desc.mutable_attrs()->Add();
    step_net_attr->set_type(paddle::framework::AttrType::INT);
    step_net_attr->set_i(2);
    step_net_attr->set_name("step_net");

    AttributeMap attrs;
    attrs["in_links"] = std::vector<int>{0};
    attrs["in_link_alias"] = std::vector<std::string>{"rnn/x"};
    attrs["out_link_alias"] = std::vector<std::string>{"rnn/h"};
    attrs["memories"] = std::vector<std::string>{"rnn/h"};
    attrs["pre_memories"] = std::vector<std::string>{"h_pre"};
    attrs["boot_memories"] = std::vector<int>{1};
    attrs["step_net"] = 2;

    LOG(INFO) << "rnn_op to init";
    rnn_op_.Init(op_desc, attrs);
    LOG(INFO) << "rnn_op finish init";
  }

  OpDesc CreateFcOpDesc() {
    OpDesc op_desc;
    op_desc.set_type("fc");
    op_desc.add_inputs("rnn/h_pre");
    op_desc.add_inputs("rnn/w");
    op_desc.add_outputs("rnn/s");
    // rnn/s = rnn/h_pre * rnn/w
    return op_desc;
  }

  OpDesc CreateAddOpDesc() {
    OpDesc op_desc;
    op_desc.set_type("add");
    op_desc.add_inputs("rnn/x");
    op_desc.add_inputs("rnn/s");
    op_desc.add_outputs("rnn/h");
    // rnn/h = rnn/x + rnn/s
    return op_desc;
  }

  void CreateStepNet() {
    LOG(INFO) << "create variable step_net";
    Variable* net_var = scope_->CreateVariable("step_net");
    NetDesc net_desc;
    net_desc.name_ = "rnn";
    net_desc.op_descs.push_back(CreateFcOpDesc());
    net_desc.op_descs.push_back(CreateAddOpDesc());
    net_var->Reset<PlainNet>(new PlainNet(net_desc));
  }

  // father scope
  std::shared_ptr<Scope> scope_;
  RecurrentOp rnn_op_;
};

// TEST_F(RecurrentOpTest, create_op) {}

TEST_F(RecurrentOpTest, Run) {
  OpContext ctx;
  ctx.scope = scope_;
  rnn_op_.Run(&ctx);
}

}  // namespace framework
}  // namespace paddle

TEST(RecurrentOp, LinkMemories) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  // create and init step scopes
  int len = 10;
  std::vector<ScopePtr> step_scopes;
  for (int i = 0; i < len; ++i) {
    auto scope = std::make_shared<Scope>();
    scope->CreateVariable("pre_h");
    auto tensor = scope->CreateVariable("h")->GetMutable<Tensor>();
    float* data = tensor->mutable_data<float>(make_ddim({15, 20}), CPUPlace());
    for (int i = 0; i < 15 * 20; ++i) {
      data[i] = rand() * (1. / (double)RAND_MAX);
    }
    step_scopes.push_back(scope);
  }

  // create MemoryAttr
  details::MemoryAttr mem_attr;
  mem_attr.pre_var = "pre_h";
  mem_attr.var = "h";
  mem_attr.boot_var = "boot_h";
  std::vector<details::MemoryAttr> memories;
  memories.push_back(mem_attr);

  for (int i = 1; i < len; ++i) {
    details::LinkMemories(step_scopes, memories, i, -1);
  }
  // check
  for (int i = 0; i < len - 1; ++i) {
    const float* a =
        step_scopes[i]->GetVariable("h")->GetMutable<Tensor>()->data<float>();
    const float* b = step_scopes[i + 1]
                         ->GetVariable("pre_h")
                         ->GetMutable<Tensor>()
                         ->data<float>();
    for (size_t i = 0; i < 15 * 20; ++i) {
      ASSERT_FLOAT_EQ(a[i], b[i]);
    }
  }

  for (int i = len - 2; i >= 0; --i) {
    details::LinkMemories(step_scopes, memories, i, 1);
  }
  // check
  for (int i = len - 2; i >= 0; --i) {
    const float* a = step_scopes[i]
                         ->GetVariable("pre_h")
                         ->GetMutable<Tensor>()
                         ->data<float>();
    const float* b = step_scopes[i + 1]
                         ->GetVariable("h")
                         ->GetMutable<Tensor>()
                         ->data<float>();
    for (size_t i = 0; i < 15 * 20; ++i) {
      ASSERT_FLOAT_EQ(a[i], b[i]);
    }
  }
}
