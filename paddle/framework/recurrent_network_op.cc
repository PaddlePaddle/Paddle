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
#include "paddle/framework/tensor.h"

#include <glog/logging.h>
#include <cstring>

namespace paddle {
namespace framework {

namespace details {

void SegmentInputs(std::vector<ScopePtr>& step_scopes) {}

void ConcatOutputs(std::vector<ScopePtr>& step_scopes) {}

void LinkMemories(std::vector<ScopePtr>& step_scopes,
                  const std::vector<details::MemoryAttr>& memories,
                  size_t step_id, int offset) {
  PADDLE_ENFORCE(step_id < step_scopes.size(),
                 "step [%d] is out of range of step scopes' size [%d]", step_id,
                 step_scopes.size());
  PADDLE_ENFORCE((static_cast<int>(step_id) + offset) >= 0 &&
                     (step_id + offset) < step_scopes.size(),
                 "the step id [%d] and offset [%d] is out of range", step_id,
                 offset);
  ScopePtr step_scope = step_scopes[step_id];
  ScopePtr linked_step_scope = step_scopes[step_id + offset];
  for (auto& attr : memories) {
    auto cur_step_pre_mem =
        step_scope->CreateVariable(attr.pre_var)->GetMutable<Tensor>();
    auto linked_step_mem =
        linked_step_scope->GetVariable(attr.var)->GetMutable<Tensor>();
    cur_step_pre_mem->ShareDataFrom<float>(*linked_step_mem);

    // TODO(qingqing) the memory of current step should be allocated in step net
    auto cur_step_mem =
        step_scope->CreateVariable(attr.var)->GetMutable<Tensor>();
    cur_step_mem->mutable_data<float>(cur_step_pre_mem->dims(),
                                      platform::CPUPlace());
  }
}

}  // namespace details

void RecurrentAlgorithm::Run(OpContext* contex) const {
  auto scope = contex->scope;

  PADDLE_ENFORCE(scope->HasVariable(net_name_), "step net is not in scope.");
  Variable* net = scope->GetVariable(net_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  LOG(INFO) << "create scopes";
  CreateScopes(scope);
  LOG(INFO) << "segment input";
  SegmentInputs(scope);

  size_t max_seq_len = GetMaxSeqLen(scope);
  LOG(INFO) << "sequence length " << max_seq_len;
  auto step_scopes = GetStepScopes(scope);
  InitMemories(step_scopes[0]);
  for (size_t step_id = 0; step_id < max_seq_len; step_id++) {
    LOG(INFO) << "run step " << step_id;
    if (step_id > 0) {
      details::LinkMemories(step_scopes, memory_attrs_, step_id, -1);
    }
    net->GetMutable<PlainNet>()->Run(step_scopes[step_id]);
  }

  LOG(INFO) << "concat outputs";
  // prepare outputs
  ConcatOutputs(scope);
}

size_t RecurrentAlgorithm::GetMaxSeqLen(ScopePtr scope) const {
  // TODO(xxx) update this function when using variable-length of sequence.
  // return Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims()[0];
  return scope->GetVariable(inlinks_[0])->GetMutable<Tensor>()->dims()[0];
}

void RecurrentAlgorithm::CreateScopes(ScopePtr scope) const {
  size_t max_seq_len = GetMaxSeqLen(scope);
  std::vector<ScopePtr>* step_scopes =
      scope->GetVariable(step_scopes_name_)
          ->GetMutable<std::vector<ScopePtr>>();
  // TODO(xxx) Only two scopes are needed for inference, this case will be
  // supported later.
  if (max_seq_len > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < max_seq_len; ++i) {
      step_scopes->push_back(std::make_shared<Scope>(scope));
    }
  }
}

void RecurrentAlgorithm::SegmentInputs(ScopePtr scope) const {
  PADDLE_ENFORCE(!inlinks_.empty(), "no in links are provided.");
  auto step_scopes = GetStepScopes(scope);
  size_t max_seq_len = GetMaxSeqLen(scope);
  for (size_t i = 0; i < inlinks_.size(); ++i) {
    // Tensor* input_tensor = Input(scope, inlinks_[i])->GetMutable<Tensor>();
    Tensor* input_tensor =
        scope->GetVariable(inlinks_[i])->GetMutable<Tensor>();
    for (size_t j = 0; j < max_seq_len; j++) {
      Variable* input_var = step_scopes[j]->CreateVariable(in_link_alias_[i]);
      Tensor* step_input_tensor = input_var->GetMutable<Tensor>();
      *step_input_tensor = input_tensor->Slice<float>(j, j + 1);
      // TODO(luotao1): use reshape function to decrease the dims of
      // step_input_tensor.
    }
  }
}

void RecurrentAlgorithm::ConcatOutputs(ScopePtr scope) const {
  auto step_scopes = GetStepScopes(scope);
  size_t max_seq_len = GetMaxSeqLen(scope);
  // TODO(luotao1): update using CopyFrom function in tensor.
  // auto dims = Input(scope, inlinks_[0])->GetMutable<Tensor>()->dims();
  auto dims = scope->GetVariable(inlinks_[0])->GetMutable<Tensor>()->dims();
  int batch_size = dims[1];
  for (size_t i = 0; i < outlinks_.size(); i++) {
    auto output_dims = step_scopes[0]
                           ->GetVariable(out_link_alias_[0])
                           ->GetMutable<Tensor>()
                           ->dims();
    int output_dim = output_dims[1];
    int length = batch_size * output_dim;
    Tensor* output_tensor =
        scope->CreateVariable(outlinks_[i])->GetMutable<Tensor>();
    float* output = output_tensor->mutable_data<float>(
        make_ddim({(int)max_seq_len, batch_size, output_dim}),
        platform::CPUPlace());
    for (size_t j = 0; j < max_seq_len; j++) {
      Variable* output_var = step_scopes[j]->GetVariable(out_link_alias_[i]);
      const float* step_output =
          output_var->GetMutable<Tensor>()->data<float>();
      std::memcpy(output + j * length, step_output, length);
    }
  }
}

void RecurrentAlgorithm::InitMemories(ScopePtr step_scope) const {
  for (auto& attr : memory_attrs_) {
    Tensor* pre_mem =
        step_scope->CreateVariable(attr.pre_var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(step_scope->HasVariable(attr.boot_var),
                   "memory [%s]'s boot variable [%s] not exists", attr.var,
                   attr.boot_var);
    Tensor* boot_mem =
        step_scope->CreateVariable(attr.boot_var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(boot_mem, "boot_tensor should be retrieved before");
    pre_mem->ShareDataFrom<float>(*boot_mem);

    // TODO(qingqing) the memory of current step should be allocated in step net
    auto cur_step_mem =
        step_scope->CreateVariable(attr.var)->GetMutable<Tensor>();
    cur_step_mem->mutable_data<float>(boot_mem->dims(), platform::CPUPlace());
  }
}

void RecurrentOp::Init(const OpDesc& op_desc, AttributeMap& attrs) {
  OperatorBase::Init(op_desc, attrs);
  // TODO(superjom) in op register will set inputs_ and outputs_
  for (const auto& item : op_desc.inputs()) {
    inputs_.emplace_back(item);
  }
  for (const auto& item : op_desc.outputs()) {
    outputs_.emplace_back(item);
  }
  // TODO(superjom) change these two copy to pointer
  alg_.inputs_ = inputs_;
  alg_.outputs_ = outputs_;

  // TODO(superjom) update following codes when variable length input
  // interfaces are added.
  alg_.net_name_ = inputs_.at(GetAttr<int>("step_net"));
  alg_.step_scopes_name_ = outputs_.back();

  // prepare inlinks
  PADDLE_ENFORCE(alg_.inlinks_.empty(), "RecurrentAlgorithm duplicate inited");
  LOG(INFO) << "set inlinks";
  for (auto id : GetAttr<std::vector<int>>("in_links")) {
    alg_.inlinks_.push_back(inputs_[id]);
  }
  auto inlink_alias = GetAttr<std::vector<std::string>>("in_link_alias");
  alg_.in_link_alias_ =
      std::vector<std::string>{inlink_alias.begin(), inlink_alias.end()};
  PADDLE_ENFORCE(alg_.inlinks_.size() == alg_.in_link_alias_.size(),
                 "in_links/in_link_alias mismatch.");

  PADDLE_ENFORCE(
      outputs_.size() > 1,
      "more than 1 output should be provided and the last is `step_scopes`");
  alg_.outlinks_ =
      std::vector<std::string>{outputs_.begin(), outputs_.end() - 1};

  auto outlink_alias = GetAttr<std::vector<std::string>>("out_link_alias");
  alg_.out_link_alias_ =
      std::vector<std::string>{outlink_alias.begin(), outlink_alias.end()};
  PADDLE_ENFORCE(alg_.outlinks_.size() == outlink_alias.size(),
                 "out_links/out_link_alias mismatch.");

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
    details::MemoryAttr mem_attr;
    mem_attr.var = memories[i];
    mem_attr.pre_var = pre_memories[i];
    mem_attr.boot_var = boot_memories[i];
    alg_.memory_attrs_.push_back(mem_attr);
    LOG(INFO) << "set memorys:\t"
              << "memory:" << mem_attr.var << "\tboot:" << mem_attr.boot_var;
  }
}

// TODO(xxx) testing when including operator.h
// class RecurrentAlgorithmProtoAndCheckerMaker : public OpProtoAndCheckerMaker
// {
//  public:
//   RecurrentAlgorithmProtoAndCheckerMaker(OpProto* proto, OpAttrChecker*
//   op_checker)
//       : OpProtoAndCheckerMaker(proto, op_checker) {
//     // AddInput("input", "input of test op"); // need to support dynamic
//     number
//     // AddOutput("output", "output of test op"); // need to support dynamic
//     number
//     AddAttr<std::std::vector<int>>("in_links", "The input link positions in
//     the all inputs.")
//         .SetDefault({0});
//     AddAttr<std::std::vector<int>>("boot_memories", "The initial memory
//     positions in the all inputs.");
//     AddAttr<int>("step_net", "The step net position in the all inputs.");
//
//     AddAttr<std::std::vector<std::string>>("in_link_alias", "The input link
//     alias in the step network.");
//     AddAttr<std::std::vector<std::string>>("out_link_alias", "The output link
//     alias in the step network.");
//     AddAttr<std::std::vector<std::string>>("memories", "The memory names.");
//     AddAttr<std::std::vector<std::string>>("pre_memories", "The
//     history/previous memory names.");
//
//     AddType("recurrent_op");
//     AddComment("This is a recurrent group operator.");
//   }
// };
//
// REGISTER_OP(recurrent_op, RecurrentAlgorithm,
// RecurrentAlgorithmProtoAndCheckerMaker);

void RecurrentGradientAlgorithm::Run(OpContext* contex) const {
  auto scope = contex->scope;
  auto step_scopes = *(scope->GetVariable(step_scopes_name_))
                          ->GetMutable<std::vector<ScopePtr>>();

  LOG(INFO) << "segment input";
  details::SegmentInputs(step_scopes);

  PADDLE_ENFORCE(scope->HasVariable(stepnet_name_),
                 "step net is not in scope.");
  Variable* net = scope->GetVariable(stepnet_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  size_t max_seq_len =
      scope->GetVariable(inlinks_[0])->GetMutable<Tensor>()->dims()[0];
  LOG(INFO) << "sequence length " << max_seq_len;

  for (size_t step_id = max_seq_len - 1; step_id > 0; --step_id) {
    LOG(INFO) << "run step " << step_id;
    if (step_id != max_seq_len - 1) {
      details::LinkMemories(step_scopes, memories_, step_id, 1);
    }
    net->GetMutable<PlainNet>()->Run(step_scopes[step_id]);
  }
  LinkBootMemoryGradients(step_scopes[0]);

  LOG(INFO) << "concat outputs";
  details::ConcatOutputs(step_scopes);
}

void RecurrentGradientAlgorithm::LinkBootMemoryGradients(
    ScopePtr step_scope) const {
  for (auto& attr : memories_) {
    Tensor* mem_g = step_scope->CreateVariable(attr.var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(mem_g, "boot_tensor should be retrieved before");

    PADDLE_ENFORCE(step_scope->HasVariable(attr.boot_var),
                   "memory [%s]'s boot variable [%s] not exists", attr.var,
                   attr.boot_var);
    Tensor* boot_mem_g =
        step_scope->CreateVariable(attr.boot_var)->GetMutable<Tensor>();
    boot_mem_g->ShareDataFrom<float>(*mem_g);
  }
}

}  // namespace framework
}  // namespace paddle
