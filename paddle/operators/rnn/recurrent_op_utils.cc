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

#include "paddle/operators/rnn/recurrent_op_utils.h"

namespace paddle {
namespace operators {
namespace rnn {

namespace f = paddle::framework;

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

void SegmentInputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<std::string>& inlinks,
                   const size_t seq_len) {
  PADDLE_ENFORCE(!inlinks.empty(), "no in links are provided.");
  for (size_t i = 0; i < inlinks.size(); ++i) {
    // global inputs
    auto input_var = step_scopes[0]->parent().FindVar(inlinks[i]);
    PADDLE_ENFORCE_NOT_NULL(input_var, "input link [%s] is not in scope.",
                            inlinks[i]);

    LoDTensor* input = input_var->GetMutable<LoDTensor>();
    f::DDim dims = input->dims();
    PADDLE_ENFORCE_EQ(static_cast<size_t>(dims[0]), seq_len,
                      "all the inputs be the same length");
    f::DDim step_dims = slice_ddim(dims, 1, dims.size());
    for (size_t j = 0; j < seq_len; j++) {
      Tensor* step_input =
          step_scopes[j]->Var(inlinks[i])->GetMutable<Tensor>();
      // The input of operators of each step is Tensor here.
      // Maybe need to modify Slice function.
      *step_input = input->Slice(j, j + 1);
      step_input->Resize(step_dims);
    }
  }
}

void ConcatOutputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<std::string>& outlinks,
                   const size_t seq_len, const platform::DeviceContext& ctx) {
  for (size_t i = 0; i < outlinks.size(); i++) {
    auto* output_var = step_scopes[0]->parent().FindVar(outlinks[i]);
    PADDLE_ENFORCE_NOT_NULL(output_var, "output link [%s] is not in scope.",
                            outlinks[i]);
    LoDTensor* output = output_var->GetMutable<LoDTensor>();

    auto* step_scope_var = step_scopes[0]->FindVar(outlinks[i]);
    PADDLE_ENFORCE_NOT_NULL(step_scope_var, "%s not in scope", outlinks[i]);
    f::DDim step_dims =
        step_scope_var->template GetMutable<LoDTensor>()->dims();
    std::vector<int64_t> dims_vec = vectorize(step_dims);
    dims_vec.insert(dims_vec.begin(), seq_len);
    output->Resize(f::make_ddim(dims_vec));
    output->mutable_data<float>(platform::CPUPlace());
    for (size_t j = 0; j < seq_len; j++) {
      LoDTensor* step_output =
          step_scopes[j]->FindVar(outlinks[i])->GetMutable<LoDTensor>();
      // TODO(luotao02) data type and platform::DeviceContext() should set
      // correctly
      (output->Slice(j, j + 1))
          .CopyFrom(*step_output, platform::CPUPlace(), ctx);
    }
  }
}

void LinkMemories(const std::vector<Scope*>& scopes,
                  const std::vector<rnn::StateAttr>& memories,
                  const size_t step_id, const int offset) {
  PADDLE_ENFORCE_LT(step_id, scopes.size(),
                    "step [%d] is out of range of step scopes' size [%d]",
                    step_id, scopes.size());
  PADDLE_ENFORCE_GE(static_cast<int>(step_id) + offset, 0,
                    "offset [%d] must be large than -[%d]", offset, step_id);
  PADDLE_ENFORCE_LT(
      step_id + offset, scopes.size(),
      "offset [%d] is out of range, it must be less than (%d - %d)", offset,
      scopes.size(), step_id);
  auto* scope = scopes[step_id];
  auto* linked_scope = scopes[step_id + offset];
  for (auto& attr : memories) {
    auto* mem = scope->FindVar(attr.pre_var)->GetMutable<LoDTensor>();
    auto* linked_mem = linked_scope->FindVar(attr.var)->GetMutable<LoDTensor>();
    mem->Resize(linked_mem->dims());
    mem->ShareDataWith(*linked_mem);
  }
}

void InitArgument(const ArgumentName& name, Argument* arg,
                  const framework::OperatorBase& op, bool is_grad) {
  arg->step_scopes =
      is_grad ? op.Input(name.step_scopes) : op.Output(name.step_scopes);
  arg->inlinks = op.Inputs(name.inlinks);
  arg->outlinks = op.Outputs(name.outlinks);

  auto& boot_memories = is_grad ? op.Outputs(name.initial_states)
                                : op.Inputs(name.initial_states);
  // attributes
  auto& memories = op.Attr<std::vector<std::string>>(name.states);
  auto& pre_memories = op.Attr<std::vector<std::string>>(name.ex_states);

  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "the size of states, initial_states don't match:%d,%d",
                 memories.size(), boot_memories.size());
  PADDLE_ENFORCE(pre_memories.size() == boot_memories.size(),
                 "the size of ex_states, initial_states don't match:%d,%d",
                 pre_memories.size(), boot_memories.size());
  PADDLE_ENFORCE(memories.size() > 0, "more than 1 states should be set");

  for (size_t i = 0; i < memories.size(); ++i) {
    rnn::StateAttr mem_attr;
    mem_attr.var = memories[i];
    mem_attr.pre_var = pre_memories[i];
    mem_attr.boot_var = boot_memories[i];
    (arg->states).push_back(mem_attr);
  }
}

}  // namespace rnn
}  // namespace operators
}  // namespace paddle
