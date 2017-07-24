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

#include "paddle/operators/recurrent_network_op.h"

#include <glog/logging.h>
#include <cstring>
#include <sstream>

#include "paddle/framework/net.h"
#include "paddle/framework/op_registry.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace operators {

namespace rnn {

void SegmentInputs(std::vector<ScopePtr>& step_scopes,
                   const std::vector<Link>& inlinks,
                   const size_t seq_len) {
  PADDLE_ENFORCE(!inlinks.empty(), "no in links are provided.");
  for (size_t i = 0; i < inlinks.size(); ++i) {
    Tensor* input =
        step_scopes[0]->GetVariable(inlinks[i].external)->GetMutable<Tensor>();
    DDim dims = input->dims();
    PADDLE_ENFORCE(static_cast<size_t>(dims[0]) == seq_len,
                   "all the inlinks must have same length");
    DDim step_dims = slice_ddim(dims, 1, dims.size());
    for (size_t j = 0; j < seq_len; j++) {
      Tensor* step_input = step_scopes[j]
                               ->CreateVariable(inlinks[i].internal)
                               ->GetMutable<Tensor>();
      *step_input = input->Slice<float>(j, j + 1);
      step_input->Resize(step_dims);
    }
  }
}

void ConcatOutputs(std::vector<ScopePtr>& step_scopes,
                   const std::vector<Link>& outlinks,
                   const size_t seq_len) {
  for (size_t i = 0; i < outlinks.size(); i++) {
    Tensor* output =
        step_scopes[0]->GetVariable(outlinks[i].external)->GetMutable<Tensor>();

    // TODO(qingiqng) remove following code after adding
    // InferShape in RecurrentGradientOp
    DDim step_dims = step_scopes[0]
                         ->GetVariable(outlinks[i].internal)
                         ->GetMutable<Tensor>()
                         ->dims();
    std::vector<int> dims_vec = vectorize(step_dims);
    dims_vec.insert(dims_vec.begin(), seq_len);
    output->mutable_data<float>(make_ddim(dims_vec), platform::CPUPlace());

    for (size_t j = 0; j < seq_len; j++) {
      Tensor* step_output = step_scopes[j]
                                ->GetVariable(outlinks[i].internal)
                                ->GetMutable<Tensor>();
      (output->Slice<float>(j, j + 1))
          .CopyFrom<float>(*step_output, platform::CPUPlace());
    }
  }
}

void LinkMemories(std::vector<ScopePtr>& scopes,
                  const std::vector<rnn::MemoryAttr>& memories,
                  size_t step_id,
                  int offset) {
  PADDLE_ENFORCE(step_id < scopes.size(),
                 "step [%d] is out of range of step scopes' size [%d]",
                 step_id,
                 scopes.size());
  PADDLE_ENFORCE(static_cast<int>(step_id) + offset >= 0,
                 "offset [%d] must be large than -[%d]",
                 offset,
                 step_id);
  PADDLE_ENFORCE(step_id + offset < scopes.size(),
                 "offset [%d] is out of range, it must be less than (%d - %d)",
                 offset,
                 scopes.size(),
                 step_id);
  ScopePtr scope = scopes[step_id];
  ScopePtr linked_scope = scopes[step_id + offset];
  for (auto& attr : memories) {
    auto mem = scope->CreateVariable(attr.pre_var)->GetMutable<Tensor>();
    // maybe share variable is better?
    auto linked_mem = linked_scope->GetVariable(attr.var)->GetMutable<Tensor>();
    mem->ShareDataWith<float>(*linked_mem);

    // TODO(qingqing) remove following code
    // the memory of current step should be allocated in step net
    auto m = scope->CreateVariable(attr.var)->GetMutable<Tensor>();
    // for unit test, as addOp and mulOp are null currently, if not
    // mutable_data, mem.data() in output will be error. We will
    // remove this line after merge the correct addOp and mulOp.
    m->mutable_data<float>(mem->dims(), platform::CPUPlace());
  }
}

void InitArgument(const ArgumentName& name,
                  Argument* arg,
                  const OperatorBase& op) {
  arg->step_net = op.Input(name.step_net);
  arg->step_scopes = op.Output(name.step_scopes);

  auto inlinks = op.Inputs(name.inlinks);
  auto inlink_alias = op.GetAttr<std::vector<std::string>>(name.inlink_alias);
  PADDLE_ENFORCE(inlinks.size() == inlink_alias.size(),
                 "the size of inlinks and inlink_alias don't match:%d,%d",
                 inlinks.size(),
                 inlink_alias.size());
  for (size_t i = 0; i < inlinks.size(); ++i) {
    rnn::Link link;
    link.external = inlinks[i];
    link.internal = inlink_alias[i];
    (arg->inlinks).push_back(link);
  }

  auto outlinks = op.Outputs(name.outlinks);
  auto outlink_alias = op.GetAttr<std::vector<std::string>>(name.outlink_alias);
  PADDLE_ENFORCE(outlinks.size() == outlink_alias.size(),
                 "the size of outlinks and outlink_alias don't match:%d,%d",
                 outlinks.size(),
                 outlink_alias.size());
  for (size_t i = 0; i < outlinks.size(); ++i) {
    rnn::Link link;
    link.external = outlinks[i];
    link.internal = outlink_alias[i];
    (arg->outlinks).push_back(link);
  }

  auto boot_memories = op.Inputs(name.boot_memories);

  // attributes
  auto memories = op.GetAttr<std::vector<std::string>>(name.memories);
  auto pre_memories = op.GetAttr<std::vector<std::string>>(name.pre_memories);

  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "the size of memories, boot_memories don't match:%d,%d",
                 memories.size(),
                 boot_memories.size());
  PADDLE_ENFORCE(pre_memories.size() == boot_memories.size(),
                 "the size of pre_memories, boot_memories don't match:%d,%d",
                 pre_memories.size(),
                 boot_memories.size());
  PADDLE_ENFORCE(memories.size() > 0, "more than 1 memories should be set");

  for (size_t i = 0; i < memories.size(); ++i) {
    rnn::MemoryAttr mem_attr;
    mem_attr.var = memories[i];
    mem_attr.pre_var = pre_memories[i];
    mem_attr.boot_var = boot_memories[i];
    (arg->memories).push_back(mem_attr);
    DLOG(INFO) << "set memorys:\t"
               << "memory:" << mem_attr.var << "\tboot:" << mem_attr.boot_var;
  }
}

}  // namespace rnn

void RecurrentAlgorithm::InferShape(const ScopePtr& scope) const {
  seq_len_ = scope->GetVariable((arg_->inlinks[0]).external)
                 ->GetMutable<Tensor>()
                 ->dims()[0];
  DLOG(INFO) << "create scopes";
  CreateScopes(scope);
  auto step_scopes = GetStepScopes(scope);

  DLOG(INFO) << "segment input";
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_);

  DLOG(INFO) << "init memories";
  InitMemories(step_scopes[0]);

  PADDLE_ENFORCE(scope->HasVariable(arg_->step_net),
                 "stepnet [%s] is not in scope.",
                 arg_->step_net);
  Variable* net = scope->GetVariable(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");
  // If the InferShape is called in OperatorBase's run function,
  // the rnn op only needs to do InferShape for the first time step
  for (size_t i = 0; i < seq_len_; i++) {
    if (i > 0) {
      rnn::LinkMemories(step_scopes, arg_->memories, i, -1);
    }
    net->GetMutable<PlainNet>()->InferShape(step_scopes[i]);
  }

  auto outlinks = arg_->outlinks;
  for (size_t i = 0; i < outlinks.size(); i++) {
    DDim step_dims = step_scopes[0]
                         ->GetVariable(outlinks[i].internal)
                         ->GetMutable<Tensor>()
                         ->dims();
    std::vector<int> dims_vec = vectorize(step_dims);
    // now only support fixed length
    dims_vec.insert(dims_vec.begin(), seq_len_);
    Tensor* output =
        step_scopes[0]->GetVariable(outlinks[i].external)->GetMutable<Tensor>();
    output->Resize(make_ddim(dims_vec));
  }
}

void RecurrentAlgorithm::Run(const ScopePtr& scope,
                             const platform::DeviceContext& dev_ctx) const {
  auto step_scopes = GetStepScopes(scope);

  Variable* net = scope->GetVariable(arg_->step_net);
  for (size_t step_id = 0; step_id < seq_len_; step_id++) {
    DLOG(INFO) << "run step " << step_id;
    // the link memory is done in InferShape
    // maybe remove following code after testing
    if (step_id > 0) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, -1);
    }
    net->GetMutable<PlainNet>()->Run(step_scopes[step_id], dev_ctx);
  }

  // prepare outputs
  DLOG(INFO) << "concat outputs";
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_);
}

std::string RecurrentAlgorithm::debug_string() const {
  std::stringstream ss;
  ss << "net_name_:\t" << arg_->step_net << '\n';
  ss << "step_scopes_name_:\t" << arg_->step_scopes << '\n';

  for (const auto& item : arg_->inlinks) {
    ss << "inlink:\t" << item.external << "\t inlink alias:" << item.internal
       << '\n';
  }

  for (const auto& item : arg_->outlinks) {
    ss << "outlink:\t" << item.external << "\t outlink alias:" << item.internal
       << '\n';
  }
  for (const auto& item : arg_->memories) {
    ss << string::Sprintf(
        "memory: %s,%s,%s\n", item.var, item.pre_var, item.boot_var);
  }
  return ss.str();
}

void RecurrentAlgorithm::CreateScopes(ScopePtr scope) const {
  // TODO(xxx) update this function when using variable-length of sequence.
  // TODO(xxx) Only two scopes are needed for inference, this case will be
  // supported later.
  std::vector<ScopePtr>* step_scopes =
      scope->GetVariable(arg_->step_scopes)
          ->GetMutable<std::vector<ScopePtr>>();
  if (seq_len_ > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < seq_len_; ++i) {
      ScopePtr step_scope = std::make_shared<Scope>(scope);

      // Now all variables in scope must be created outside of op.
      auto net_op = scope->GetVariable(arg_->step_net)->GetMutable<PlainNet>();
      for (auto& input : net_op->inputs_) {
        step_scope->CreateVariable(input);
      }
      for (auto& output : net_op->outputs_) {
        step_scope->CreateVariable(output);
      }

      step_scopes->push_back(std::make_shared<Scope>(step_scope));
    }
  }
}

void RecurrentAlgorithm::InitMemories(ScopePtr step_scope) const {
  for (auto& attr : arg_->memories) {
    Tensor* pre_mem =
        step_scope->CreateVariable(attr.pre_var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(step_scope->HasVariable(attr.boot_var),
                   "memory [%s]'s boot variable [%s] not exists",
                   attr.var,
                   attr.boot_var);
    Tensor* boot_mem =
        step_scope->CreateVariable(attr.boot_var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(boot_mem != nullptr,
                   "boot_tensor should be retrieved before");
    pre_mem->ShareDataWith<float>(*boot_mem);

    // TODO(qingqing) remove following code
    // the memory of current step should be allocated in step net
    // here for unit test
    auto cur_step_mem =
        step_scope->CreateVariable(attr.var)->GetMutable<Tensor>();
    cur_step_mem->mutable_data<float>(boot_mem->dims(), platform::CPUPlace());
  }
}

const rnn::ArgumentName RecurrentOp::arg_name{"step_net",
                                              "step_scopes",
                                              "inlinks",
                                              "outlinks",
                                              "inlink_alias",
                                              "outlink_alias",
                                              "memories",
                                              "pre_memories",
                                              "boot_memories"};

const rnn::ArgumentName RecurrentGradientOp::arg_name{"step_net",
                                                      "step_scopes",
                                                      "outlink@grad",
                                                      "inlink@grad",
                                                      "inlink_alias",
                                                      "outlink_alias",
                                                      "memories",
                                                      "pre_memories",
                                                      "boot_memories@grad"};

void RecurrentOp::Init() {
  OperatorBase::Init();
  std::unique_ptr<rnn::Argument> arg(new rnn::Argument());

  rnn::InitArgument(arg_name, arg.get(), *this);

  alg_.Init(std::move(arg));

  DLOG(INFO) << alg_.debug_string();
}

/*
 * Op definition of RNNOp
 */
class RecurrentAlgorithmProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
public:
  RecurrentAlgorithmProtoAndCheckerMaker(OpProto* proto,
                                         OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    const auto& name = RecurrentOp::arg_name;
    // inputs and outputs stored in proto
    AddInputs(name.inlinks,
              "the input that need to be segmented for each step.");
    AddInputs(name.boot_memories, "variables to initialize memories.");
    AddInput(name.step_net, "network shared by all steps.");

    AddOutputs(name.outlinks,
               "the output that need to concated for all steps.");
    AddOutput(name.step_scopes, "step scopes");

    // Attributes stored in AttributeMap
    AddAttr<std::vector<std::string>>(name.inlink_alias, "alias of inlinks");
    AddAttr<std::vector<std::string>>(name.outlink_alias, "alias of outlinks");
    AddAttr<std::vector<std::string>>(name.pre_memories,
                                      "names of pre-memories");
    AddAttr<std::vector<std::string>>(name.memories, "names of memories");

    AddComment("This is a recurrent group operator.");
  }
};

void RecurrentGradientAlgorithm::Run(
    const ScopePtr& scope, const platform::DeviceContext& dev_ctx) const {
  auto step_scopes = *(scope->GetVariable(arg_->step_scopes))
                          ->GetMutable<std::vector<ScopePtr>>();
  size_t seq_len = scope->GetVariable((arg_->inlinks[0]).external)
                       ->GetMutable<Tensor>()
                       ->dims()[0];

  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len);

  PADDLE_ENFORCE(scope->HasVariable(arg_->step_net),
                 "step net is not in scope.");
  Variable* net = scope->GetVariable(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");

  for (int step_id = seq_len - 1; step_id >= 0; --step_id) {
    LOG(INFO) << step_id;
    DLOG(INFO) << "run step " << step_id;
    if (static_cast<size_t>(step_id) != seq_len - 1) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, 1);
    }
    net->GetMutable<PlainNet>()->Run(step_scopes[step_id], dev_ctx);
  }
  LinkBootMemoryGradients(step_scopes[0]);

  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len);
}

void RecurrentGradientAlgorithm::LinkBootMemoryGradients(
    ScopePtr step_scope) const {
  for (auto& attr : arg_->memories) {
    Tensor* mem_g = step_scope->CreateVariable(attr.var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(mem_g != nullptr, "boot_tensor should be retrieved before");

    PADDLE_ENFORCE(step_scope->HasVariable(attr.boot_var),
                   "memory [%s]'s boot variable [%s] not exists",
                   attr.var,
                   attr.boot_var);
    Tensor* boot_mem_g =
        step_scope->CreateVariable(attr.boot_var)->GetMutable<Tensor>();
    boot_mem_g->ShareDataWith<float>(*mem_g);
  }
}

// TODO(Superjom) implement this after op's members move to rnn namespace
void RecurrentGradientOp::Init() {
  OperatorBase::Init();
  std::unique_ptr<rnn::Argument> arg(new rnn::Argument());

  rnn::InitArgument(arg_name, arg.get(), *this);

  alg_.Init(std::move(arg));
}

void RecurrentGradientAlgorithm::InferShape(const ScopePtr& scope) const {
  auto step_scopes = *(scope->GetVariable(arg_->step_scopes))
                          ->GetMutable<std::vector<ScopePtr>>();
  seq_len_ = scope->GetVariable((arg_->inlinks[0]).external)
                 ->GetMutable<Tensor>()
                 ->dims()[0];
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_);

  PADDLE_ENFORCE(scope->HasVariable(arg_->step_net),
                 "step net is not in scope.");
  Variable* net = scope->GetVariable(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");

  for (int step_id = seq_len_ - 1; step_id >= 0; --step_id) {
    if (static_cast<size_t>(step_id) != seq_len_ - 1) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, 1);
    }

    net->GetMutable<PlainNet>()->InferShape(step_scopes[step_id]);
  }

  auto outlinks = arg_->outlinks;
  for (size_t i = 0; i < outlinks.size(); i++) {
    DDim step_dims = step_scopes[0]
                         ->GetVariable(outlinks[i].internal)
                         ->GetMutable<Tensor>()
                         ->dims();
    std::vector<int> dims_vec = vectorize(step_dims);
    // now only support fixed length
    dims_vec.insert(dims_vec.begin(), seq_len_);
    Tensor* output =
        step_scopes[0]->GetVariable(outlinks[i].external)->GetMutable<Tensor>();
    output->Resize(make_ddim(dims_vec));
  }

  LinkBootMemoryGradients(step_scopes[0]);
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP(recurrent_op,
            ::paddle::operators::RecurrentOp,
            ::paddle::operators::RecurrentAlgorithmProtoAndCheckerMaker);
