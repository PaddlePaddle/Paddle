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

#include "paddle/framework/op_registry.h"
#include "paddle/framework/recurrent_network_op.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

// fake op implementations
namespace fake {
class FcOp : public OperatorBase {
 public:
  FcOp(const OpDesc& desc) {}

  virtual void InferShape(const ScopePtr& scope) const override {
    for (const auto& output : outputs_) {
      LOG(INFO) << "fc [" << name_ << "]"
                << " create output variable [" << output << "]";
      scope->CreateVariable(output);
    }
  }

  virtual void Run(const ScopePtr& scope,
                   const platform::DeviceContext& dev_ctx) const override {
    LOG(INFO) << "run fc op";
    for (const auto& input : inputs_) {
      PADDLE_ENFORCE(scope->HasVariable(input),
                     "no input variable [%s] exists");
      LOG(INFO) << "fc [" << name_ << "] read input [" << input << "]";
    }
    for (const auto& output : outputs_) {
      PADDLE_ENFORCE(scope->HasVariable(output),
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

  virtual void InferShape(const ScopePtr& scope) const override {
    for (const auto& output : outputs_) {
      LOG(INFO) << "add [" << name_ << "]"
                << " create output variable [" << output << "]";
      scope->CreateVariable(output);
    }
  }

  virtual void Run(const ScopePtr& scope,
                   const platform::DeviceContext& dev_ctx) const override {
    LOG(INFO) << "run add op";
    for (const auto& input : inputs_) {
      PADDLE_ENFORCE(scope->HasVariable(input),
                     "no input variable [%s] exists");
      LOG(INFO) << "add [" << name_ << "] read input [" << input << "]";
    }
    for (const auto& output : outputs_) {
      PADDLE_ENFORCE(scope->HasVariable(output),
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
    for (auto inlink : std::vector<std::string>{"x", "x0", "x1", "h"}) {
      Variable* x = scope_->CreateVariable(inlink);
      DDim dims = make_ddim(std::vector<int>{
          10 /*sent size*/, 20 /*batch size*/, 30 /*input dim*/});
      x->GetMutable<Tensor>()->mutable_data<float>(dims, platform::CPUPlace());
    }

    LOG(INFO) << "create global variable w";
    Variable* w = scope_->CreateVariable("rnn/w");
    w->GetMutable<Tensor>()->mutable_data<float>(
        make_ddim(std::vector<int>{30, 30}), platform::CPUPlace());

    for (auto boot : std::vector<std::string>{"x_boot", "h_boot"}) {
      LOG(INFO) << "create global variable " << boot;
      Variable* h_boot = scope_->CreateVariable(boot);
      h_boot->GetMutable<Tensor>()->mutable_data<float>(
          make_ddim(std::vector<int>{20 /*batch size*/, 30 /*input dim*/}),
          platform::CPUPlace());
    }

    LOG(INFO) << "create variable step_scopes";
    scope_->CreateVariable("step_scopes");

    LOG(INFO) << "create variable h";
    scope_->CreateVariable("h");
  }

  void CreateRNNOp() {
    OpDesc op_desc;

    op_desc.set_type("recurrent_op");
    // inlinks 0
    op_desc.add_inputs("x");
    op_desc.add_inputs("x0");
    op_desc.add_inputs("x1");
    // memories 3
    op_desc.add_inputs("rnn/x");
    op_desc.add_inputs("rnn/h");
    // pre-memories 5
    op_desc.add_inputs("rnn/x@pre");
    op_desc.add_inputs("rnn/h@pre");
    // boot_memories 7
    op_desc.add_inputs("x_boot");
    op_desc.add_inputs("h_boot");
    // inlink_alias 9
    op_desc.add_inputs("x@alias");
    op_desc.add_inputs("x0@alias");
    op_desc.add_inputs("x1@alias");
    // step net 12
    op_desc.add_inputs("step_net");
    // outlinks 0
    op_desc.add_outputs("h");
    // outlink_alias 1
    op_desc.add_outputs("h@alias");
    // step scopes 2
    op_desc.add_outputs("step_scopes");

    auto _input_format = std::vector<int>{
        0,  // in_link
        3,  // memories
        5,  // pre-memories
        7,  // boot_memories
        9,  // input_alias
        12  // step_net
    };
    auto input_format = op_desc.add_attrs();
    input_format->set_name("input_format");
    input_format->set_type(paddle::framework::AttrType::INTS);
    for (auto i : _input_format) {
      input_format->add_ints(i);
    }

    auto _output_format = std::vector<int>{0, 1, 2};
    auto output_format = op_desc.add_attrs();
    output_format->set_name("output_format");
    output_format->set_type(paddle::framework::AttrType::INTS);
    for (auto i : _output_format) {
      output_format->add_ints(i);
    }

    LOG(INFO) << "rnn_op to init";
    // set inputs, outputs and attrs
    // TODO(superjom) use CreateOp instead
    // for (const auto& item : op_desc.inputs()) {
    //   rnn_op_.inputs_.emplace_back(item);
    // }
    // for (const auto& item : op_desc.outputs()) {
    //   rnn_op_.outputs_.emplace_back(item);
    // }
    rnn_op_ = OpRegistry::CreateOp(op_desc);

    // rnn_op_.Init();
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
  OperatorPtr rnn_op_;
};

// TEST_F(RecurrentOpTest, create_op) {}

TEST_F(RecurrentOpTest, Run) {
  platform::CPUDeviceContext ctx;
  rnn_op_->Run(scope_, ctx);
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
