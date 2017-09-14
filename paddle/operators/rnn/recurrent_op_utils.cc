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
                   const std::vector<Link>& inlinks, const size_t seq_len,
                   bool infer_shape_mode) {
  PADDLE_ENFORCE(!inlinks.empty(), "no in links are provided.");
  for (size_t i = 0; i < inlinks.size(); ++i) {
    auto input_var = step_scopes[0]->FindVar(inlinks[i].external);
    PADDLE_ENFORCE(input_var != nullptr, "input link [%s] is not in scope.",
                   inlinks[i].external);

    LoDTensor* input = input_var->GetMutable<LoDTensor>();
    f::DDim dims = input->dims();
    PADDLE_ENFORCE(static_cast<size_t>(dims[0]) == seq_len,
                   "all the inlinks must have same length");
    f::DDim step_dims = slice_ddim(dims, 1, dims.size());
    for (size_t j = 0; j < seq_len; j++) {
      Tensor* step_input =
          step_scopes[j]->NewVar(inlinks[i].internal)->GetMutable<Tensor>();
      if (!infer_shape_mode) {
        // The input of operators of each step is Tensor here.
        // Maybe need to modify Slice function.
        *step_input = input->Slice<float>(j, j + 1);
      }
      step_input->Resize(step_dims);
    }
  }
}

void ConcatOutputs(const std::vector<Scope*>& step_scopes,
                   const std::vector<Link>& outlinks, const size_t seq_len,
                   bool infer_shape_mode) {
  for (size_t i = 0; i < outlinks.size(); i++) {
    auto output_var = step_scopes[0]->FindVar(outlinks[i].external);
    PADDLE_ENFORCE(output_var != nullptr, "output link [%s] is not in scope.",
                   outlinks[i].external);
    LoDTensor* output = output_var->GetMutable<LoDTensor>();

    if (infer_shape_mode) {
      auto step_scope_var = step_scopes[0]->FindVar(outlinks[i].internal);
      PADDLE_ENFORCE(step_scope_var != nullptr, "%s not in scope",
                     outlinks[i].internal);
      f::DDim step_dims =
          step_scope_var->template GetMutable<LoDTensor>()->dims();
      std::vector<int64_t> dims_vec = vectorize(step_dims);
      dims_vec.insert(dims_vec.begin(), seq_len);
      output->Resize(f::make_ddim(dims_vec));
    } else {
      output->mutable_data<float>(platform::CPUPlace());
      for (size_t j = 0; j < seq_len; j++) {
        LoDTensor* step_output = step_scopes[j]
                                     ->FindVar(outlinks[i].internal)
                                     ->GetMutable<LoDTensor>();
        // TODO(luotao02) data type and platform::DeviceContext() should set
        // correctly
        (output->Slice<float>(j, j + 1))
            .CopyFrom<float>(*step_output, platform::CPUPlace());
      }
    }
  }
}

void LinkMemories(const std::vector<Scope*>& scopes,
                  const std::vector<rnn::MemoryAttr>& memories,
                  const size_t step_id, const int offset,
                  bool infer_shape_mode) {
  PADDLE_ENFORCE_LT(step_id, scopes.size(),
                    "step [%d] is out of range of step scopes' size [%d]",
                    step_id, scopes.size());
  PADDLE_ENFORCE_GE(static_cast<int>(step_id) + offset, 0,
                    "offset [%d] must be large than -[%d]", offset, step_id);
  PADDLE_ENFORCE_LT(
      step_id + offset, scopes.size(),
      "offset [%d] is out of range, it must be less than (%d - %d)", offset,
      scopes.size(), step_id);
  auto scope = scopes[step_id];
  auto linked_scope = scopes[step_id + offset];
  for (auto& attr : memories) {
    auto mem = scope->FindVar(attr.pre_var)->GetMutable<LoDTensor>();
    auto linked_mem = linked_scope->FindVar(attr.var)->GetMutable<LoDTensor>();
    if (infer_shape_mode) {
      mem->Resize(linked_mem->dims());
    } else {
      mem->ShareDataWith<float>(*linked_mem);
    }
  }
}

void InitArgument(const ArgumentName& name, Argument* arg,
                  const framework::OperatorBase& op) {
  arg->step_scopes = op.Output(name.step_scopes);

  auto inlinks = op.Inputs(name.inlinks);
  auto inlink_alias = op.Attr<std::vector<std::string>>(name.inlink_alias);
  PADDLE_ENFORCE(inlinks.size() == inlink_alias.size(),
                 "the size of inlinks and inlink_alias don't match:%d,%d",
                 inlinks.size(), inlink_alias.size());
  for (size_t i = 0; i < inlinks.size(); ++i) {
    rnn::Link link;
    link.external = inlinks[i];
    link.internal = inlink_alias[i];
    (arg->inlinks).push_back(link);
  }

  auto outlinks = op.Outputs(name.outlinks);
  auto outlink_alias = op.Attr<std::vector<std::string>>(name.outlink_alias);
  PADDLE_ENFORCE(outlinks.size() == outlink_alias.size(),
                 "the size of outlinks and outlink_alias don't match:%d,%d",
                 outlinks.size(), outlink_alias.size());
  for (size_t i = 0; i < outlinks.size(); ++i) {
    rnn::Link link;
    link.external = outlinks[i];
    link.internal = outlink_alias[i];
    (arg->outlinks).push_back(link);
  }

  auto boot_memories = op.Inputs(name.boot_memories);

  // attributes
  auto memories = op.Attr<std::vector<std::string>>(name.memories);
  auto pre_memories = op.Attr<std::vector<std::string>>(name.pre_memories);

  PADDLE_ENFORCE(memories.size() == boot_memories.size(),
                 "the size of memories, boot_memories don't match:%d,%d",
                 memories.size(), boot_memories.size());
  PADDLE_ENFORCE(pre_memories.size() == boot_memories.size(),
                 "the size of pre_memories, boot_memories don't match:%d,%d",
                 pre_memories.size(), boot_memories.size());
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
}  // namespace operators
}  // namespace paddle
