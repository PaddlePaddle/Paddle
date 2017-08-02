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

void SegmentInputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<Link>& inlinks,
                   const size_t seq_len) {
  PADDLE_ENFORCE(!inlinks.empty(), "no in links are provided.");
  for (size_t i = 0; i < inlinks.size(); ++i) {
    Tensor* input =
        step_scopes[0]->FindVar(inlinks[i].external)->GetMutable<Tensor>();
    DDim dims = input->dims();
    PADDLE_ENFORCE(static_cast<size_t>(dims[0]) == seq_len,
                   "all the inlinks must have same length");
    DDim step_dims = slice_ddim(dims, 1, dims.size());
    for (size_t j = 0; j < seq_len; j++) {
      Tensor* step_input =
          step_scopes[j]->NewVar(inlinks[i].internal)->GetMutable<Tensor>();
      *step_input = input->Slice<float>(j, j + 1);
      step_input->Resize(step_dims);
    }
  }
}

void ConcatOutputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<Link>& outlinks,
                   const size_t seq_len) {
  for (size_t i = 0; i < outlinks.size(); i++) {
    Tensor* output =
        step_scopes[0]->FindVar(outlinks[i].external)->GetMutable<Tensor>();

    // TODO(qingiqng) remove following code after adding
    // InferShape in RecurrentGradientOp
    DDim step_dims = step_scopes[0]
                         ->FindVar(outlinks[i].internal)
                         ->GetMutable<Tensor>()
                         ->dims();
    std::vector<int> dims_vec = vectorize(step_dims);
    dims_vec.insert(dims_vec.begin(), seq_len);
    output->mutable_data<float>(make_ddim(dims_vec), platform::CPUPlace());

    for (size_t j = 0; j < seq_len; j++) {
      Tensor* step_output =
          step_scopes[j]->FindVar(outlinks[i].internal)->GetMutable<Tensor>();
      // TODO(luotao02) data type and platform::DeviceContext() should set
      // correctly
      (output->Slice<float>(j, j + 1))
          .CopyFrom<float>(*step_output, platform::CPUPlace());
    }
  }
}

void LinkMemories(const std::vector<Scope*>& scopes,
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
  auto scope = scopes[step_id];
  auto linked_scope = scopes[step_id + offset];
  for (auto& attr : memories) {
    auto mem = scope->NewVar(attr.pre_var)->GetMutable<Tensor>();
    // maybe share variable is better?
    auto linked_mem = linked_scope->FindVar(attr.var)->GetMutable<Tensor>();
    mem->ShareDataWith<float>(*linked_mem);

    // TODO(qingqing) remove following code
    // the memory of current step should be allocated in step net
    auto m = scope->NewVar(attr.var)->GetMutable<Tensor>();
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
  }
}

}  // namespace rnn

void RecurrentAlgorithm::InferShape(const Scope& scope) const {
  seq_len_ = scope.FindVar((arg_->inlinks[0]).external)
                 ->GetMutable<Tensor>()
                 ->dims()[0];
  CreateScopes(scope);
  auto step_scopes = GetStepScopes(scope);

  // SegmentInputs is called in InferShape. The input must hold memory in
  // SegmentInputs. But the other op only set dimension for the output in
  // InferShape. That's a problem. Wether the RNN op needs InferShape or not?
  // Wether the following functions (SegmentInputs, InitMemories, ...) need
  // to rewrite for RNN op?
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_);

  InitMemories(step_scopes[0]);

  PADDLE_ENFORCE(scope.FindVar(arg_->step_net) != nullptr,
                 "stepnet [%s] is not in scope.",
                 arg_->step_net);
  Variable* net = scope.FindVar(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");
  // If the InferShape is called in OperatorBase's run function,
  // the rnn op only needs to do InferShape for the first time step
  for (size_t i = 0; i < seq_len_; i++) {
    if (i > 0) {
      rnn::LinkMemories(step_scopes, arg_->memories, i, -1);
    }
    net->GetMutable<NetOp>()->InferShape(*step_scopes[i]);
  }

  auto outlinks = arg_->outlinks;
  for (size_t i = 0; i < outlinks.size(); i++) {
    DDim step_dims = step_scopes[0]
                         ->FindVar(outlinks[i].internal)
                         ->GetMutable<Tensor>()
                         ->dims();
    std::vector<int> dims_vec = vectorize(step_dims);
    // now only support fixed length
    dims_vec.insert(dims_vec.begin(), seq_len_);
    Tensor* output =
        step_scopes[0]->FindVar(outlinks[i].external)->GetMutable<Tensor>();
    output->Resize(make_ddim(dims_vec));
  }
}

void RecurrentAlgorithm::Run(const Scope& scope,
                             const platform::DeviceContext& dev_ctx) const {
  auto step_scopes = GetStepScopes(scope);

  Variable* net = scope.FindVar(arg_->step_net);
  for (size_t step_id = 0; step_id < seq_len_; step_id++) {
    // the link memory is done in InferShape
    // maybe remove following code after testing
    if (step_id > 0) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, -1);
    }
    net->GetMutable<NetOp>()->Run(*step_scopes[step_id], dev_ctx);
  }

  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_);
}

void RecurrentAlgorithm::CreateScopes(const Scope& scope) const {
  // TODO(xxx) Only two scopes are needed for inference, this case will be
  // supported later.
  auto step_scopes =
      scope.FindVar(arg_->step_scopes)->GetMutable<std::vector<Scope*>>();

  if (seq_len_ > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < seq_len_; ++i) {
      auto& step_scope = scope.NewScope();

      // Now all variables in scope must be created outside of op.
      auto net_op = scope.FindVar(arg_->step_net)->GetMutable<NetOp>();
      for (auto& input : net_op->inputs_) {
        if (!step_scope.FindVar(input)) step_scope.NewVar(input);
      }
      for (auto& output : net_op->outputs_) {
        step_scope.NewVar(output);
      }

      step_scopes->emplace_back(&step_scope);
    }
  }
}

void RecurrentAlgorithm::InitMemories(Scope* step_scope) const {
  for (auto& attr : arg_->memories) {
    Tensor* pre_mem = step_scope->NewVar(attr.pre_var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(step_scope->FindVar(attr.boot_var) != nullptr,
                   "memory [%s]'s boot variable [%s] not exists",
                   attr.var,
                   attr.boot_var);
    Tensor* boot_mem = step_scope->FindVar(attr.boot_var)->GetMutable<Tensor>();
    pre_mem->ShareDataWith<float>(*boot_mem);

    // TODO(qingqing) remove following code
    // the memory of current step should be allocated in step net
    // here for unit test
    auto cur_step_mem = step_scope->NewVar(attr.var)->GetMutable<Tensor>();
    cur_step_mem->mutable_data<float>(boot_mem->dims(), platform::CPUPlace());
  }
}

const rnn::ArgumentName RecurrentOp::kArgName{"step_net",
                                              "step_scopes",
                                              "inlinks",
                                              "outlinks",
                                              "inlink_alias",
                                              "outlink_alias",
                                              "memories",
                                              "pre_memories",
                                              "boot_memories"};

const rnn::ArgumentName RecurrentGradientOp::kArgName{"step_net",
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
  rnn::InitArgument(kArgName, arg.get(), *this);
  alg_.Init(std::move(arg));
}

class RecurrentAlgorithmProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
public:
  RecurrentAlgorithmProtoAndCheckerMaker(OpProto* proto,
                                         OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    const auto& name = RecurrentOp::kArgName;
    // inputs and outputs stored in proto
    AddInput(name.inlinks, "the input that need to be segmented for each step.")
        .SetMultiple();
    AddInput(name.boot_memories, "variables to initialize memories.")
        .SetMultiple();
    AddInput(name.step_net, "network shared by all steps.");

    AddOutput(name.outlinks, "the output that need to concated for all steps.")
        .SetMultiple();
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
    const Scope& scope, const platform::DeviceContext& dev_ctx) const {
  auto step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_);
  PADDLE_ENFORCE(scope.FindVar(arg_->step_net) != nullptr,
                 "step net is not in scope.");
  Variable* net = scope.FindVar(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");
  for (int step_id = seq_len_ - 1; step_id >= 0; --step_id) {
    if (static_cast<size_t>(step_id) != seq_len_ - 1) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, 1);
    }
    net->GetMutable<NetOp>()->Run(*step_scopes[step_id], dev_ctx);
  }
  LinkBootMemoryGradients(step_scopes[0]);
  rnn::ConcatOutputs(step_scopes, arg_->outlinks, seq_len_);
}

void RecurrentGradientAlgorithm::LinkBootMemoryGradients(
    Scope* step_scope) const {
  for (auto& attr : arg_->memories) {
    Tensor* mem_grad = step_scope->NewVar(attr.var)->GetMutable<Tensor>();
    PADDLE_ENFORCE(mem_grad != nullptr,
                   "boot_tensor should be retrieved before");
    PADDLE_ENFORCE(step_scope->FindVar(attr.boot_var) != nullptr,
                   "memory [%s]'s boot variable [%s] not exists",
                   attr.var,
                   attr.boot_var);
    Tensor* boot_mem_grad =
        step_scope->NewVar(attr.boot_var)->GetMutable<Tensor>();
    boot_mem_grad->ShareDataWith<float>(*mem_grad);
  }
}

void RecurrentGradientAlgorithm::InferShape(const Scope& scope) const {
  seq_len_ = scope.FindVar((arg_->inlinks[0]).external)
                 ->GetMutable<Tensor>()
                 ->dims()[0];
  auto step_scopes = GetStepScopes(scope);
  rnn::SegmentInputs(step_scopes, arg_->inlinks, seq_len_);

  PADDLE_ENFORCE(scope.FindVar(arg_->step_net) != nullptr,
                 "step net is not in scope.");
  Variable* net = scope.FindVar(arg_->step_net);
  PADDLE_ENFORCE(net != nullptr, "failed to get step net");

  for (int step_id = seq_len_ - 1; step_id >= 0; --step_id) {
    if (static_cast<size_t>(step_id) != seq_len_ - 1) {
      rnn::LinkMemories(step_scopes, arg_->memories, step_id, 1);
    }
    net->GetMutable<NetOp>()->InferShape(*step_scopes[step_id]);
  }

  auto outlinks = arg_->outlinks;
  for (size_t i = 0; i < outlinks.size(); i++) {
    DDim step_dims = step_scopes[0]
                         ->FindVar(outlinks[i].internal)
                         ->GetMutable<Tensor>()
                         ->dims();
    std::vector<int> dims_vec = vectorize(step_dims);
    // now only support fixed length
    dims_vec.insert(dims_vec.begin(), seq_len_);
    Tensor* output =
        step_scopes[0]->FindVar(outlinks[i].external)->GetMutable<Tensor>();
    output->Resize(make_ddim(dims_vec));
  }
  LinkBootMemoryGradients(step_scopes[0]);
}

void RecurrentGradientOp::Init() {
  OperatorBase::Init();
  std::unique_ptr<rnn::Argument> arg(new rnn::Argument());
  rnn::InitArgument(kArgName, arg.get(), *this);
  alg_.Init(std::move(arg));
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP(recurrent_op,
            paddle::operators::RecurrentOp,
            paddle::operators::RecurrentAlgorithmProtoAndCheckerMaker);
