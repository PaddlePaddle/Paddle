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

void RecurrentOp::Run(OpContext* contex) const {
  auto scope = contex->scope;

  PADDLE_ENFORCE(scope->HasVariable(net_name_), "step net is not in scope.");
  Variable* net = scope->GetVariable(net_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  LOG(INFO) << "create scopes";
  CreateScopes(scope);
  LOG(INFO) << "segment input";
  SegmentInputs(scope);

  Variable* step_scopes = scope->GetVariable(step_scopes_name_);
  PADDLE_ENFORCE(step_scopes, "failed to get step scopes");
  // forward
  auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
  size_t seq_len = dims[0];
  LOG(INFO) << "sequence length " << seq_len;
  auto& scopes = *step_scopes->GetMutable<std::vector<ScopePtr>>();
  for (size_t step_id = 0; step_id < seq_len; step_id++) {
    LOG(INFO) << "run step " << step_id;
    ScopePtr step_scope = scopes[step_id];
    // TODO replace memorys' copy with reference
    LinkMemories(scope, scopes, step_id);

    net->GetMutable<PlainNet>()->Run(step_scope);
  }

  LOG(INFO) << "concat outputs";
  // prepare outputs
  ConcatOutputs(scope);
}

void RecurrentOp::Init(const OpDesc& op_desc, AttributeMap& attrs) {
  OperatorBase::Init(op_desc, attrs);

  // set original inputs
  for (const std::string& input : op_desc.inputs()) {
    LOG(INFO) << "set input " << input;
    inputs_.push_back(input);
  }
  // set original outputs
  for (const std::string& output : op_desc.outputs()) {
    LOG(INFO) << "set output " << output;
    outputs_.push_back(output);
  }

  name_ = op_desc.name();
  net_name_ = inputs_.at(GetAttr<int>("step_net"));
  step_scopes_name_ = outputs_.back();

  // prepare inlinks
  PADDLE_ENFORCE(inlinks_.empty(), "RecurrentOp duplicate inited");
  LOG(INFO) << "set inlinks";
  for (auto id : GetAttr<std::vector<int>>("real_inputs")) {
    inlinks_.push_back(id);
  }
  PADDLE_ENFORCE(
      outputs_.size() > 1,
      "more than 1 output should be provided and the last is `step_scopes`");
  outlinks_ = std::vector<std::string>{outputs_.begin(), outputs_.end() - 1};

  // set memories
  auto memories = GetAttr<std::vector<std::string>>("memories");
  auto pre_memories = GetAttr<std::vector<std::string>>("pre_memories");
  PADDLE_ENFORCE(memories.size() == pre_memories.size(),
                 "The size of memories and pre_memories doesn't match: %d,%d.",
                 memories.size(), pre_memories.size());

  std::vector<std::string> boot_memories;
  LOG(INFO) << "set boot_memories";
  for (auto id : GetAttr<std::vector<int>>("boot_memories")) {
    boot_memories.push_back(inputs_[id]);
  }
  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "the size of memories and boot_memories doesn't match: %d,%d",
                 memories.size(), boot_memories.size());
  for (size_t i = 0; i < memories.size(); ++i) {
    MemoryAttr mem_attr;
    mem_attr.var = memories[i];
    mem_attr.pre_var = pre_memories[i];
    mem_attr.boot_var = boot_memories[i];
    memory_attrs_.push_back(mem_attr);
    LOG(INFO) << "set memorys:\t"
              << "memory:" << mem_attr.var << "\tboot:" << mem_attr.boot_var;
  }
}

void RecurrentOp::CreateScopes(ScopePtr scope) const {
  auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
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

// void RecurrentOp::CreateStepNet(ScopePtr scope) const {
//   Variable* var = scope->CreateVariable(net_name_);
//   auto step_net = GetAttr<std::string>("step_net");
//   // get the step net proto from the string.
//   // PADDLE_ENFORCE(
//   //   google::protobuf::TextFormat::ParseFromString(step_net,
//   //   &step_net_desc_));
//   // var->Reset<PlainNet>(new PlainNet(step_net_desc_));
//   // this is a fake net, it will be rewrite after the network has been
//   merged.
//   NetDesc desc;
//   desc.name_ = "rnn_step_net";
//   var->Reset<PlainNet>(new PlainNet(desc));
// }

void RecurrentOp::SegmentInputs(ScopePtr scope) const {
  PADDLE_ENFORCE(!inlinks_.empty(), "no real inputs are provided.");
  auto input_alias = GetAttr<std::vector<std::string>>("input_alias");
  PADDLE_ENFORCE(inlinks_.size() == input_alias.size(),
                 "real_inputs/input_alias mismatch.");

  Variable* scopes_var = scope->GetVariable(step_scopes_name_);
  auto& step_scopes = *scopes_var->GetMutable<std::vector<ScopePtr>>();
  auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
  int seq_len = dims[0];
  for (size_t i = 0; i < inlinks_.size(); ++i) {
    Tensor* scope_input_tensor =
        Input(scope, inlinks_[i])->GetMutable<Tensor>();
    for (int j = 0; j < seq_len; j++) {
      Variable* input_var = step_scopes[j]->CreateVariable(input_alias[i]);
      Tensor* step_input_tensor = input_var->GetMutable<Tensor>();
      *step_input_tensor = scope_input_tensor->Slice(j, j + 1);
    }
  }
}

void RecurrentOp::ConcatOutputs(ScopePtr scope) const {
  auto output_alias = GetAttr<std::vector<std::string>>("output_alias");
  PADDLE_ENFORCE(outlinks_.size() == output_alias.size(),
                 "output/output_alias mismatch.");

  Variable* scopes_var = scope->GetVariable(step_scopes_name_);
  auto& step_scopes = *scopes_var->GetMutable<std::vector<ScopePtr>>();
  auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
  int seq_len = dims[0];
  int batch_size = dims[1];
  for (size_t i = 0; i < outlinks_.size(); i++) {
    auto output_dims = step_scopes[0]
                           ->GetVariable(output_alias[0])
                           ->GetMutable<Tensor>()
                           ->dims();
    int output_dim = output_dims[1];
    int length = batch_size * output_dim;
    Tensor* output_tensor =
        scope->CreateVariable(outlinks_[i])->GetMutable<Tensor>();
    float* output = output_tensor->mutable_data<float>(
        make_ddim({seq_len, batch_size, output_dim}), platform::CPUPlace());
    for (int j = 0; j < seq_len; j++) {
      Variable* output_var = step_scopes[j]->GetVariable(output_alias[i]);
      const float* step_output =
          output_var->GetMutable<Tensor>()->data<float>();
      std::memcpy(output + j * length, step_output, length);
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

    // TODO the memory of current step should be allocaled in step net ?
    Tensor* cur_memory =
        step_scopes[step]->CreateVariable(attr.var)->GetMutable<Tensor>();
    cur_memory->mutable_data<float>(attr.dims, platform::CPUPlace());

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
