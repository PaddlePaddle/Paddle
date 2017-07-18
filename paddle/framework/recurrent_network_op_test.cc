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
    // set inputs, outputs and attrs
    // TODO(superjom) use CreateOp instead
    for (const auto& item : op_desc.inputs()) {
      rnn_op_.inputs_.emplace_back(item);
    }
    for (const auto& item : op_desc.outputs()) {
      rnn_op_.outputs_.emplace_back(item);
    }
    rnn_op_.attrs_ = attrs;
    rnn_op_.Init();
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
  platform::CPUDeviceContext ctx;
  rnn_op_.Run(scope_, ctx);
}

class RecurrentGradientAlgorithmTest : public ::testing::Test {
 protected:
  virtual void SetUp() override {
    CreateGlobalVariables();
    CreateStepScopes();
    CreateStepNet();
    CreateRNNGradientAlgorithm();

    // segment inputs
    SegmentInputs();
    // link forward memories
    LinkeMemories();
  }

  virtual void TearDown() override {}

  void CreateGlobalVariables() {
    scope_ = std::make_shared<Scope>();
    // inputs: x
    LOG(INFO) << "create global variable x";
    Variable* x = scope_->CreateVariable("x");
    DDim dims =
        make_ddim({10 /*sent size*/, 20 /*batch size*/, 30 /*input dim*/});
    x->GetMutable<Tensor>()->mutable_data<float>(dims, platform::CPUPlace());
    // inputs: h_boot
    LOG(INFO) << "create global variable h_boot";
    Variable* h_boot = scope_->CreateVariable("h_boot");
    h_boot->GetMutable<Tensor>()->mutable_data<float>(
        make_ddim({20 /*batch size*/, 30 /*input dim*/}), platform::CPUPlace());
    // inputs: w
    LOG(INFO) << "create global variable w";
    Variable* w = scope_->CreateVariable("rnn/w");
    w->GetMutable<Tensor>()->mutable_data<float>(make_ddim({30, 30}),
                                                 platform::CPUPlace());
    // inputs: h_grad
    LOG(INFO) << "create variable h_grad";
    Variable* dh = scope_->CreateVariable("h_grad");
    dh->GetMutable<Tensor>()->mutable_data<float>(make_ddim({10, 20, 30}),
                                                  platform::CPUPlace());
    // inputs: step_scopes
    LOG(INFO) << "create variable step_scopes";
    scope_->CreateVariable("step_scopes");
    // inputs: step_net
    LOG(INFO) << "create variable step_net";
    scope_->CreateVariable("step_net");
    // outputs: w_grad
    LOG(INFO) << "create global variable w_grad";
    scope_->CreateVariable("rnn/w_grad");
    // outputs: x_grad
    LOG(INFO) << "create global variable x_grad";
    scope_->CreateVariable("x_grad");
    // outputs: h_boot_grad
    LOG(INFO) << "create global variable h_boot_grad";
    scope_->CreateVariable("h_boot_grad");
  }

  void CreateStepScopes() {
    std::vector<ScopePtr>* step_scopes =
        scope_->GetVariable("step_scopes")->GetMutable<std::vector<ScopePtr>>();
    for (int i = 0; i < 10; ++i) {
      auto scope = std::make_shared<Scope>(scope_);
      auto pre_t = scope->CreateVariable("rnn/pre_h")->GetMutable<Tensor>();
      pre_t->mutable_data<float>(make_ddim({20, 30}), platform::CPUPlace());
      auto tensor = scope->CreateVariable("rnn/h")->GetMutable<Tensor>();
      tensor->mutable_data<float>(make_ddim({20, 30}), platform::CPUPlace());

      // for unit test of ConcatOutputs
      auto xg = scope->CreateVariable("rnn/x_grad")->GetMutable<Tensor>();
      xg->mutable_data<float>(make_ddim({20, 30}), platform::CPUPlace());

      step_scopes->push_back(scope);
    }

    // last time step
    auto g = (*step_scopes)[9]
                 ->CreateVariable("rnn/h_pre_grad")
                 ->GetMutable<Tensor>();
    g->mutable_data<float>(make_ddim({20, 30}), platform::CPUPlace());
  }

  void CreateRNNGradientAlgorithm() {
    AttributeMap attrs;
    attrs["step_net"] = "step_net";
    attrs["step_scopes"] = "step_scopes";
    attrs["in_links"] = std::vector<std::string>{"h_grad"};
    attrs["in_link_alias"] = std::vector<std::string>{"rnn/h_grad"};
    attrs["out_links"] = std::vector<std::string>{"x_grad"};
    attrs["out_link_alias"] = std::vector<std::string>{"rnn/x_grad"};
    attrs["memories"] = std::vector<std::string>{"rnn/h_pre_grad"};
    attrs["pre_memories"] = std::vector<std::string>{"rnn/h_grad"};
    attrs["boot_memories"] = std::vector<std::string>{"h_boot_grad"};
    rnn_grad_algo_.Init(attrs);
  }

  OpDesc CreateFcGradientOpDesc() {
    OpDesc op_desc;
    op_desc.set_type("fc");  // use fc op for test
    op_desc.add_inputs("rnn/s_grad");
    op_desc.add_inputs("rnn/h_pre");
    op_desc.add_inputs("rnn/w");
    op_desc.add_outputs("rnn/h_pre_grad");
    op_desc.add_outputs("rnn/w_grad");
    // rnn/h_pre_grad = rnn/s_grad * trans(rnn/w)
    // rnn/w_grad = trans(rnn/h_pre) * rnn/s_grad
    return op_desc;
  }

  OpDesc CreateAddGradientOpDesc() {
    OpDesc op_desc;
    op_desc.set_type("add");  // use add op for test
    op_desc.add_inputs("rnn/h_grad");
    op_desc.add_outputs("rnn/x_grad");
    op_desc.add_outputs("rnn/s_grad");
    // rnn/x_grad = rnn/h_grad
    // rnn/s_grad = rnn/h_grad
    return op_desc;
  }

  void CreateStepNet() {
    LOG(INFO) << "create variable step_net";
    Variable* net_var = scope_->CreateVariable("step_net");
    NetDesc net_desc;
    net_desc.name_ = "rnn_gradient";
    net_desc.op_descs.push_back(CreateFcGradientOpDesc());
    net_desc.op_descs.push_back(CreateAddGradientOpDesc());
    net_var->Reset<PlainNet>(new PlainNet(net_desc));
  }

  void SegmentInputs() {
    LOG(INFO) << "segment inputs";
    std::vector<std::string> inlinks = {"x"};
    std::vector<std::string> inlinks_alias = {"rnn/x"};
    std::vector<ScopePtr>* step_scopes =
        scope_->GetVariable("step_scopes")->GetMutable<std::vector<ScopePtr>>();
    details::SegmentInputs(*step_scopes, inlinks, inlinks_alias);
  }

  void LinkeMemories() {
    LOG(INFO) << "link memories";
    details::MemoryAttr mem_attr;
    mem_attr.pre_var = "rnn/h_pre";
    mem_attr.var = "rnn/h";
    mem_attr.boot_var = "boot_h";
    std::vector<details::MemoryAttr> memories;
    memories.push_back(mem_attr);
    std::vector<ScopePtr>* step_scopes =
        scope_->GetVariable("step_scopes")->GetMutable<std::vector<ScopePtr>>();
    for (int i = 1; i < 10; ++i) {
      details::LinkMemories(*step_scopes, memories, i, -1);
    }
  }

  std::shared_ptr<Scope> scope_;
  RecurrentGradientAlgorithm rnn_grad_algo_;
};

TEST_F(RecurrentGradientAlgorithmTest, Run) {
  platform::CPUDeviceContext ctx;
  rnn_grad_algo_.Run(scope_, ctx);
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
