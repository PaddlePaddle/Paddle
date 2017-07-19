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

#include <glog/logging.h>
#include <cstring>
#include <sstream>

#include "paddle/framework/enforce.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/recurrent_network_op.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

namespace details {

void SegmentInputs(std::vector<ScopePtr>& step_scopes,
                   const std::vector<std::string>& inlinks,
                   const std::vector<std::string>& inlinks_alias) {
  PADDLE_ENFORCE(!inlinks.empty(), "no in links are provided.");
  for (size_t i = 0; i < inlinks.size(); ++i) {
    Tensor* input =
        step_scopes[0]->GetVariable(inlinks[i])->GetMutable<Tensor>();
    DDim dims = input->dims();
    DDim step_dims = slice_ddim(dims, 1, dims.size());
    for (size_t j = 0; j < step_scopes.size(); j++) {
      Tensor* step_input = step_scopes[j]
                               ->CreateVariable(inlinks_alias[i])
                               ->GetMutable<Tensor>();
      *step_input = input->Slice<float>(j, j + 1);
      step_input->set_dims(step_dims);
    }
  }
}

void ConcatOutputs(std::vector<ScopePtr>& step_scopes,
                   const std::vector<std::string>& outlinks,
                   const std::vector<std::string>& outlinks_alias) {
  for (size_t i = 0; i < outlinks.size(); i++) {
    DDim step_dims = step_scopes[0]
                         ->GetVariable(outlinks_alias[i])
                         ->GetMutable<Tensor>()
                         ->dims();
    std::vector<int> dims_vec = vectorize(step_dims);
    dims_vec.insert(dims_vec.begin(), step_scopes.size());

    Tensor* output =
        step_scopes[0]->CreateVariable(outlinks[i])->GetMutable<Tensor>();
    output->mutable_data<double>(make_ddim(dims_vec), platform::CPUPlace());

    for (size_t j = 0; j < step_scopes.size(); j++) {
      Tensor* step_output = step_scopes[j]
                                ->CreateVariable(outlinks_alias[i])
                                ->GetMutable<Tensor>();
      (output->Slice<float>(j, j + 1))
          .CopyFrom<float>(*step_output, platform::CPUPlace());
    }
  }
}

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

void RecurrentAlgorithm::Run(const ScopePtr& scope,
                             const platform::DeviceContext& dev_ctx) const {
  PADDLE_ENFORCE(scope->HasVariable(net_name_), "stepnet [%s] is not in scope.",
                 net_name_);
  Variable* net = scope->GetVariable(net_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  DLOG(INFO) << "create scopes";
  CreateScopes(scope);
  auto step_scopes = GetStepScopes(scope);

  DLOG(INFO) << "segment input";
  details::SegmentInputs(step_scopes, inlinks_, inlink_alias_);

  InitMemories(step_scopes[0]);
  for (size_t step_id = 0; step_id < step_scopes.size(); step_id++) {
    DLOG(INFO) << "run step " << step_id;
    if (step_id > 0) {
      details::LinkMemories(step_scopes, memory_attrs_, step_id, -1);
    }
    net->GetMutable<PlainNet>()->Run(step_scopes[step_id], dev_ctx);
  }

  // prepare outputs
  DLOG(INFO) << "concat outputs";
  details::ConcatOutputs(step_scopes, outlinks_, outlink_alias_);
}

std::string RecurrentAlgorithm::debug_string() const {
  std::stringstream ss;
  ss << "net_name_:\t" << net_name_ << '\n';
  ss << "step_scopes_name_:\t" << step_scopes_name_ << '\n';

  for (const auto& item : inlinks_) {
    ss << "inlink:\t" << item << '\n';
  }
  for (const auto& item : outlinks_) {
    ss << "outlink:\t" << item << '\n';
  }
  for (const auto& item : inlink_alias_) {
    ss << "inlink alias:\t" << item << '\n';
  }
  for (const auto& item : outlink_alias_) {
    ss << "outlink alias:\t" << item << '\n';
  }
  for (const auto& item : memory_attrs_) {
    ss << string::Sprintf("memory: %s,%s,%s\n", item.var, item.pre_var,
                          item.boot_var);
  }
  return ss.str();
}

void RecurrentAlgorithm::CreateScopes(ScopePtr scope) const {
  // TODO(xxx) update this function when using variable-length of sequence.
  size_t max_seq_len =
      scope->GetVariable(inlinks_[0])->GetMutable<Tensor>()->dims()[0];
  DLOG(INFO) << "sequence length " << max_seq_len;
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

    // TODO(qingqing) the memory of current step should be allocated in step
    // net
    auto cur_step_mem =
        step_scope->CreateVariable(attr.var)->GetMutable<Tensor>();
    cur_step_mem->mutable_data<float>(boot_mem->dims(), platform::CPUPlace());
  }
}

void RecurrentOp::Init() {
  OperatorBase::Init();
  alg_.inputs_ = inputs_;
  alg_.outputs_ = outputs_;

  alg_.net_name_ = Input("step_net");
  alg_.step_scopes_name_ = Output("step_scopes");

  alg_.inlinks_ = Inputs("inlinks");
  alg_.inlink_alias_ = GetAttr<std::vector<std::string>>("inlink_alias");

  alg_.outlinks_ = Outputs("outlinks");
  alg_.outlink_alias_ = GetAttr<std::vector<std::string>>("outlink_alias");
  auto boot_memories = Inputs("boot_memories");

  // attributes
  auto memories = GetAttr<std::vector<std::string>>("memories");
  auto pre_memories = GetAttr<std::vector<std::string>>("pre_memories");

  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "the size of memories, pre_memories don't match:%d,%d",
                 memories.size(), pre_memories.size());
  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "the size of memories, boot_memories don't match:%d,%d",
                 memories.size(), boot_memories.size());
  PADDLE_ENFORCE(memories.size() > 0, "more than 1 memories should be set");

  for (size_t i = 0; i < memories.size(); ++i) {
    details::MemoryAttr mem_attr;
    mem_attr.var = memories[i];
    mem_attr.pre_var = pre_memories[i];
    mem_attr.boot_var = boot_memories[i];
    alg_.memory_attrs_.push_back(mem_attr);
    DLOG(INFO) << "set memorys:\t"
               << "memory:" << mem_attr.var << "\tboot:" << mem_attr.boot_var;
  }

  DLOG(INFO) << alg_.debug_string();
}

class RecurrentAlgorithmProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  RecurrentAlgorithmProtoAndCheckerMaker(OpProto* proto,
                                         OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInputs("inlinks", "the input that need to be segmented for each step.");
    AddInputs("boot_memories", "variables to initialize memories.");

    AddInput("step_net", "network shared by all steps.");

    AddOutputs("outlinks", "the output that need to concated for all steps.");
    AddOutput("step_scopes", "step scopes");

    AddAttr<std::vector<std::string>>("inlink_alias", "alias of inlinks");
    AddAttr<std::vector<std::string>>("outlink_alias", "alias of outlinks");
    AddAttr<std::vector<std::string>>("pre_memories", "names of pre-memories");
    AddAttr<std::vector<std::string>>("memories", "names of memories");

    AddComment("This is a recurrent group operator.");
  }
};

void RecurrentGradientAlgorithm::Run(
    const ScopePtr& scope, const platform::DeviceContext& dev_ctx) const {
  auto step_scopes = *(scope->GetVariable(step_scopes_name_))
                          ->GetMutable<std::vector<ScopePtr>>();

  DLOG(INFO) << "segment input";
  details::SegmentInputs(step_scopes, inlinks_, inlink_alias_);

  PADDLE_ENFORCE(scope->HasVariable(stepnet_name_),
                 "step net is not in scope.");
  Variable* net = scope->GetVariable(stepnet_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  size_t max_seq_len =
      scope->GetVariable(inlinks_[0])->GetMutable<Tensor>()->dims()[0];
  DLOG(INFO) << "sequence length " << max_seq_len;

  for (int step_id = max_seq_len - 1; step_id >= 0; --step_id) {
    DLOG(INFO) << "run step " << step_id;
    if (static_cast<size_t>(step_id) != max_seq_len - 1) {
      details::LinkMemories(step_scopes, memories_, step_id, 1);
    }
    net->GetMutable<PlainNet>()->Run(step_scopes[step_id], dev_ctx);
  }
  LinkBootMemoryGradients(step_scopes[0]);

  DLOG(INFO) << "concat outputs";
  details::ConcatOutputs(step_scopes, outlinks_, outlink_alias_);
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

void RecurrentGradientAlgorithm::Init(AttributeMap& attrs) {
  stepnet_name_ = boost::get<std::string>(attrs.at("step_net"));
  step_scopes_name_ = boost::get<std::string>(attrs.at("step_scopes"));

  auto inlinks = boost::get<std::vector<std::string>>(attrs.at("in_links"));
  inlinks_ = std::vector<std::string>{inlinks.begin(), inlinks.end()};

  auto inlink_alias =
      boost::get<std::vector<std::string>>(attrs.at("in_link_alias"));
  inlink_alias_ =
      std::vector<std::string>{inlink_alias.begin(), inlink_alias.end()};
  PADDLE_ENFORCE(inlinks_.size() == inlink_alias_.size(),
                 "in_links/in_link_alias mismatch.");

  auto outlinks = boost::get<std::vector<std::string>>(attrs.at("out_links"));
  outlinks_ = std::vector<std::string>{outlinks.begin(), outlinks.end()};

  auto outlink_alias =
      boost::get<std::vector<std::string>>(attrs.at("out_link_alias"));
  outlink_alias_ =
      std::vector<std::string>{outlink_alias.begin(), outlink_alias.end()};
  PADDLE_ENFORCE(outlinks_.size() == outlink_alias_.size(),
                 "out_links/out_link_alias mismatch.");

  // set memories
  auto memories = boost::get<std::vector<std::string>>(attrs.at("memories"));
  auto pre_memories =
      boost::get<std::vector<std::string>>(attrs.at("pre_memories"));
  auto boot_memories =
      boost::get<std::vector<std::string>>(attrs.at("boot_memories"));

  PADDLE_ENFORCE(memories.size() == pre_memories.size(),
                 "The size of memories and pre_memories doesn't match: %d,%d.",
                 memories.size(), pre_memories.size());
  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "the size of memories and boot_memories doesn't match: %d,%d",
                 memories.size(), boot_memories.size());
  for (size_t i = 0; i < memories.size(); ++i) {
    details::MemoryAttr mem_attr;
    mem_attr.var = memories[i];
    mem_attr.pre_var = pre_memories[i];
    mem_attr.boot_var = boot_memories[i];
    memories_.push_back(mem_attr);
    DLOG(INFO) << "set memorys:\t"
               << "memory:" << mem_attr.var << "\tboot:" << mem_attr.boot_var;
  }
}

}  // namespace framework
}  // namespace paddle

REGISTER_OP(recurrent_op, ::paddle::framework::RecurrentOp,
            ::paddle::framework::RecurrentAlgorithmProtoAndCheckerMaker);
