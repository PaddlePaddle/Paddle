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

#include "paddle/framework/recurrent_network_op.h"

#include <glog/logging.h>
#include <cstring>

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
}  // namespace fake

void PlainNet::AddOp(const OpDesc& desc) {
  if (desc.type() == "fc") {
    ops_.emplace_back(new fake::FcOp(desc));
  }
}

void RecurrentOp::Run(OpContext* contex) const {
  auto scope = contex->scope;

  if (!scope->HasVariable(net_name_)) {
    CreateStepNet(scope);
  }
  Variable* net = scope->GetVariable(net_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  CreateScopes(scope);
  SegmentInputs(scope);

  Variable* step_scopes = scope->GetVariable(step_scopes_name_);
  PADDLE_ENFORCE(step_scopes, "failed to get step scopes");
  // forward
  auto dims = Input(scope, 0)->GetMutable<Tensor>()->dims();
  size_t seq_len = dims[0];
  auto& scopes = *step_scopes->GetMutable<std::vector<ScopePtr>>();
  for (size_t step_id = 0; step_id < seq_len; step_id++) {
    ScopePtr step_scope = scopes[step_id];
    // TODO replace memorys' copy with reference
    LinkMemories(scope, scopes, step_id);

    net->GetMutable<PlainNet>()->Run(step_scope);
  }

  // prepare outputs
  ConcateOutputs(scope);
}

void RecurrentOp::Init(const OpDesc& op_desc, AttributeMap& attrs) {
  OperatorBase::Init(op_desc, attrs);
  name_ = op_desc.name();
  net_name_ = op_desc.name() + "_net";
  step_scopes_name_ = op_desc.name() + "_step_scopes";
  auto memories = GetAttr<std::vector<std::string>>("memories");
  auto boot_memories = GetAttr<std::vector<std::string>>("boot_memories");
  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "The size of memories and boot_memories is mismatched.");
  // set memories
  for (size_t i = 0; i < memories.size(); ++i) {
    MemoryAttr mem_attr;
    mem_attr.var = memories[i];
    mem_attr.boot_var = boot_memories[i];
    memory_attrs_.push_back(mem_attr);
    LOG(INFO) << "set memorys:\t"
              << "memory:" << mem_attr.var << "\tboot:" << mem_attr.boot_var;
  }

  // set inputs
  for (const std::string& input : op_desc.inputs()) {
    LOG(INFO) << "set input " << input;
    inputs_.push_back(input);
  }
  // set outputs
  for (const std::string& output : op_desc.outputs()) {
    LOG(INFO) << "set output " << output;
    outputs_.push_back(output);
  }
}

void RecurrentOp::CreateScopes(ScopePtr scope) const {
  LOG(INFO) << "create scopes";
  auto dims = Input(scope, 0)->GetMutable<Tensor>()->dims();
  size_t seq_len = dims[0];
  Variable* scopes_var = scope->GetVariable(step_scopes_name_);
  auto step_scopes = scopes_var->GetMutable<std::vector<ScopePtr>>();
  // TODO Only two scopes are needed for inference, this case will be
  // supported later.
  if (seq_len > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < seq_len; ++i) {
      step_scopes->push_back(std::make_shared<Scope>(scope));
    }
  }
}

void RecurrentOp::CreateStepNet(ScopePtr scope) const {
  Variable* var = scope->CreateVariable(net_name_);
  auto step_net = GetAttr<std::string>("step_net");
  // get the step net proto from the string.
  // PADDLE_ENFORCE(
  //   google::protobuf::TextFormat::ParseFromString(step_net,
  //   &step_net_desc_));
  // var->Reset<PlainNet>(new PlainNet(step_net_desc_));
  // this is a fake net, it will be rewrite after the network has been merged.
  NetDesc desc;
  desc.name_ = "rnn_step_net";
  var->Reset<PlainNet>(new PlainNet(desc));
  // TODO add op descs
}

void RecurrentOp::SegmentInputs(ScopePtr scope) const {
  Variable* scopes_var = scope->CreateVariable(step_scopes_name_);
  auto& step_scopes = *scopes_var->GetMutable<std::vector<Scope*>>();

  auto dims = Input(scope, 0)->GetMutable<Tensor>()->dims();
  int seq_len = dims[0];
  int batch_size = dims[1];
  int dim = dims[2];
  int length = batch_size * dim;
  for (size_t i = 0; i < inputs_.size(); i++) {
    const float* scope_input =
        Input(scope, i)->GetMutable<Tensor>()->data<float>();
    for (int j = 0; j < seq_len; j++) {
      std::string name =
          name_ + "@input_" + inputs_[i] + "@step_" + std::to_string(j);
      Variable* input_var = step_scopes[j]->CreateVariable(name);
      Tensor* step_input_tensor = input_var->GetMutable<Tensor>();
      float* step_input = step_input_tensor->mutable_data<float>(
          make_ddim({1, batch_size, dim}), platform::CPUPlace());
      std::memcpy(step_input, scope_input + j * length, length);
    }
  }
}

void RecurrentOp::LinkMemories(ScopePtr scope,
                               std::vector<ScopePtr>& step_scopes,
                               size_t step) const {
  PADDLE_ENFORCE(step < step_scopes.size(),
                 "step [%d] out of range of step scopes' size [%d]", step,
                 step_scopes.size());
  auto step_scope = step_scopes[step];
  // copy boot memory
  for (auto& attr : memory_attrs_) {
    Tensor* boot_tensor{nullptr};
    if (step == 0) {
      PADDLE_ENFORCE(scope->HasVariable(attr.boot_var),
                     "memory [%s]'s boot variable [%s] not exists", attr.var,
                     attr.boot_var);
      // update memory's ddim
      boot_tensor = scope->CreateVariable(attr.boot_var)->GetMutable<Tensor>();
      attr.dims = boot_tensor->dims();
    }
    Variable* memory_var = step_scope->CreateVariable(attr.pre_var);

    // copy from boot memory
    // TODO support more device
    // TODO mutable_data is currently invalid
    float* memory_tensor_val =
        memory_var->GetMutable<Tensor>()->mutable_data<float>(
            attr.dims, platform::CPUPlace());
    if (step == 0) {
      PADDLE_ENFORCE(boot_tensor, "boot_tensor should be retrieved before");
      // copy from boot memory
      std::memcpy(memory_tensor_val, boot_tensor->data<float>(),
                  product(attr.dims));
    } else {
      // copy from previous step scope's memory to this scope's
      // `pre - memory`
      Tensor* pre_step_memory =
          step_scopes[step - 1]->GetVariable(attr.var)->GetMutable<Tensor>();
      std::memcpy(memory_tensor_val, pre_step_memory->data<float>(),
                  product(attr.dims));
    }
  }
}

}  // namespace framework
}  // namespace paddle
