// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/enforce.h"

COMMON_DECLARE_string(tensor_operants_mode);

namespace paddle {
namespace prim {

/*
  This functor class is responsible for creating the gradient ops for the given
  operator fwd_op_. After it is called (through operator()), the pairs of
  (gradient variable, corresponding input variable of fwd_op_) will be added to
  grad_to_var. If an input variable of fwd_op_ is contained in no_grad_set, its
  gradient variable will be ignored or kEmptyVarName depending on the template
  argument DropEmptyIG in the derived classes.
 */

class CompositeGradOpMakerBase {
 public:
  explicit CompositeGradOpMakerBase(
      const framework::OpDesc& fwd_op,
      const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var,
      const framework::BlockDesc* original_block,
      const std::vector<framework::BlockDesc*>& grad_block =
          std::vector<framework::BlockDesc*>())
      : fwd_op_(fwd_op),
        no_grad_set_(no_grad_set),
        grad_to_var_(grad_to_var),
        original_block_(original_block),
        acting_program_(framework::ProgramDesc()),
        grad_block_(grad_block) {
    // TODO(jiabin): This should always execute by one thread...
    VLOG(6) << "Constructing Composite Grad func for " << fwd_op_.Type()
            << "_grad ";
    FLAGS_tensor_operants_mode = "static";
    StaticCompositeContext::Instance().SetBlock(
        acting_program_.MutableBlock(0));
  }

  virtual ~CompositeGradOpMakerBase() = default;

  virtual std::vector<std::unique_ptr<framework::OpDesc>> operator()() {
    VLOG(3) << "Running Composite Grad func for " << fwd_op_.Type() << "_grad ";
    this->Apply();
    std::vector<std::unique_ptr<framework::OpDesc>> ops;
    // TODO(jiabin): Support multiple blocks later
    for (auto* op : StaticCompositeContext::Instance().GetBlock()->AllOps()) {
      ops.emplace_back(new framework::OpDesc(*op));
      ops.back()->ResetBlock();
    }
    return ops;
  }

  virtual void Apply() = 0;

  Tensor GetSingleForwardOutput(const std::string& name) {
    framework::VarDesc* out_desc = this->SingleForwardOutput(name);
    Tensor out = Tensor(std::make_shared<DescTensor>(out_desc));
    return out;
  }

  Tensor GetSingleForwardInput(const std::string& name) {
    Tensor input =
        Tensor(std::make_shared<DescTensor>(this->SingleForwardInput(name)));
    return input;
  }

  Tensor GetSingleOutputGrad(const std::string& name) {
    Tensor output_grad =
        Tensor(std::make_shared<DescTensor>(this->SingleOutputGrad(name)));
    return output_grad;
  }

  // TODO(Ruting): modify name to GetNullableSingleInputGrad after Large-scale
  // development
  Tensor GetSingleInputGrad(const std::string& name) {
    framework::VarDesc* input_grad_desc = this->SingleInputGrad(name);
    if (!input_grad_desc) return Tensor();
    Tensor input_grad = Tensor(std::make_shared<DescTensor>(input_grad_desc));
    return input_grad;
  }

  paddle::optional<Tensor> GetOptionalSingleForwardOutput(
      const std::string& name) {
    paddle::optional<Tensor> output_opt;
    if (fwd_op_.Outputs().find(name) != fwd_op_.Outputs().end()) {
      framework::VarDesc* output_desc = this->SingleForwardOutput(name);
      if (!output_desc) return output_opt;
      Tensor output = Tensor(std::make_shared<DescTensor>(output_desc));
      output_opt = paddle::make_optional<Tensor>(output);
    }
    return output_opt;
  }

  paddle::optional<Tensor> GetOptionalSingleForwardInput(
      const std::string& name) {
    paddle::optional<Tensor> input_opt;
    if (fwd_op_.Inputs().find(name) != fwd_op_.Inputs().end()) {
      framework::VarDesc* input_desc = this->SingleForwardInput(name);
      if (!input_desc) return input_opt;
      Tensor input = Tensor(std::make_shared<DescTensor>(input_desc));
      input_opt = paddle::make_optional<Tensor>(input);
    }
    return input_opt;
  }

  paddle::optional<Tensor> GetOptionalSingleOutputGrad(
      const std::string& name) {
    paddle::optional<Tensor> output_grad_opt;
    if (fwd_op_.Outputs().find(name) != fwd_op_.Outputs().end()) {
      framework::VarDesc* output_grad_desc = this->SingleOutputGrad(name);
      if (!output_grad_desc) return output_grad_opt;
      Tensor output_grad =
          Tensor(std::make_shared<DescTensor>(output_grad_desc));
      output_grad_opt = paddle::make_optional<Tensor>(output_grad);
    }
    return output_grad_opt;
  }

  std::vector<Tensor> GetMultiForwardOutput(const std::string& name) {
    std::vector<Tensor> outputs;
    std::vector<framework::VarDesc*> outputs_descs =
        this->MultiForwardOutput(name);
    outputs.reserve(outputs_descs.size());
    for (const auto& output_desc : outputs_descs) {
      outputs.emplace_back(Tensor(std::make_shared<DescTensor>(output_desc)));
    }
    return outputs;
  }

  std::vector<Tensor> GetMultiForwardInput(const std::string& name) {
    std::vector<Tensor> inputs;
    std::vector<framework::VarDesc*> inputs_descs =
        this->MultiForwardInput(name);
    inputs.reserve(inputs_descs.size());
    for (const auto& input_desc : inputs_descs) {
      inputs.emplace_back(Tensor(std::make_shared<DescTensor>(input_desc)));
    }
    return inputs;
  }

  std::vector<Tensor> GetMultiOutputGrad(const std::string& name) {
    std::vector<Tensor> outputs_grads;
    std::vector<framework::VarDesc*> outputs_grads_descs =
        this->MultiOutputGrad(name);
    outputs_grads.reserve(outputs_grads_descs.size());
    for (const auto& output_grad_desc : outputs_grads_descs) {
      outputs_grads.emplace_back(
          Tensor(std::make_shared<DescTensor>(output_grad_desc)));
    }
    return outputs_grads;
  }

  std::vector<Tensor> GetMultiInputGrad(const std::string& name) {
    std::vector<Tensor> inputs_grads;
    std::vector<framework::VarDesc*> inputs_grads_descs =
        this->MultiInputGrad(name);
    inputs_grads.reserve(inputs_grads_descs.size());
    for (const auto& input_grad_desc : inputs_grads_descs) {
      if (input_grad_desc) {
        inputs_grads.emplace_back(
            Tensor(std::make_shared<DescTensor>(input_grad_desc)));
      } else {
        inputs_grads.emplace_back(Tensor());
      }
    }
    return inputs_grads;
  }

  paddle::optional<std::vector<Tensor>> GetOptionalMultiForwardOutput(
      const std::string& name) {
    paddle::optional<std::vector<Tensor>> outputs_opt;
    std::vector<framework::VarDesc*> outputs_descs =
        this->MultiForwardOutput(name);
    if ((outputs_descs.empty())) {
      return outputs_opt;
    }
    std::vector<Tensor> outputs;
    outputs.reserve(outputs_descs.size());
    for (const auto& output_desc : outputs_descs) {
      if (output_desc) {
        outputs.emplace_back(
            Tensor(Tensor(std::make_shared<DescTensor>(output_desc))));
      } else {
        outputs.emplace_back(Tensor(Tensor()));
      }
    }
    outputs_opt = paddle::make_optional<std::vector<Tensor>>(outputs);
    return outputs_opt;
  }

  paddle::optional<std::vector<Tensor>> GetOptionalMultiForwardInput(
      const std::string& name) {
    paddle::optional<std::vector<Tensor>> inputs_opt;
    std::vector<framework::VarDesc*> inputs_descs =
        this->MultiForwardInput(name);
    if ((inputs_descs.empty())) {
      return inputs_opt;
    }
    std::vector<Tensor> inputs;
    inputs.reserve(inputs_descs.size());
    for (const auto& input_desc : inputs_descs) {
      if (input_desc) {
        inputs.emplace_back(
            Tensor(Tensor(std::make_shared<DescTensor>(input_desc))));
      } else {
        inputs.emplace_back(Tensor(Tensor()));
      }
    }
    inputs_opt = paddle::make_optional<std::vector<Tensor>>(inputs);
    return inputs_opt;
  }

  paddle::optional<std::vector<Tensor>> GetOptionalMultiOutputGrad(
      const std::string& name) {
    paddle::optional<std::vector<Tensor>> outputs_grads_opt;
    std::vector<framework::VarDesc*> outputs_grads_descs =
        this->MultiOutputGrad(name);
    if ((outputs_grads_descs.empty())) {
      return outputs_grads_opt;
    }
    std::vector<Tensor> outputs_grads;
    outputs_grads.reserve(outputs_grads_descs.size());
    for (const auto& output_grad_desc : outputs_grads_descs) {
      if (output_grad_desc) {
        outputs_grads.emplace_back(
            Tensor(Tensor(std::make_shared<DescTensor>(output_grad_desc))));
      } else {
        outputs_grads.emplace_back(Tensor(Tensor()));
      }
    }
    outputs_grads_opt =
        paddle::make_optional<std::vector<Tensor>>(outputs_grads);
    return outputs_grads_opt;
  }

  Tensor* GetOutputPtr(Tensor* input) {
    if (input->defined()) return input;
    return nullptr;
  }

  std::vector<Tensor*> GetOutputPtr(const std::vector<Tensor*>& inputs) {
    std::vector<Tensor*> output_ptrs;
    output_ptrs.reserve(inputs.size());
    for (const auto& input : inputs) {
      if (input->defined())
        output_ptrs.emplace_back(input);
      else
        output_ptrs.emplace_back(nullptr);
    }
    return output_ptrs;
  }

  std::string GetOutputName(const Tensor& output) {
    if (!output.defined()) return framework::kEmptyVarName;
    return static_cast<DescTensor*>(output.impl().get())->Name();
  }

  std::vector<std::string> GetOutputName(const std::vector<Tensor>& outputs) {
    std::vector<std::string> out_names;
    out_names.reserve(outputs.size());
    for (const auto& output : outputs) {
      if (!output.defined())
        out_names.emplace_back(framework::kEmptyVarName);
      else
        out_names.emplace_back(
            static_cast<DescTensor*>(output.impl().get())->Name());
    }
    return out_names;
  }

 protected:
  void CopyVarFromOrig(const std::string& name) const {
    VLOG(6) << "Copy Var: " << name << "from block: " << original_block_
            << " to block: " << StaticCompositeContext::Instance().GetBlock();
    framework::VarDesc* original_var = original_block_->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        original_var,
        common::errors::InvalidArgument(
            "Can't find var: %s in block %s", name, original_block_));
    *StaticCompositeContext::Instance().GetBlock()->Var(name) = *original_var;
  }

  framework::VarDesc* SingleInputGrad(const std::string& name,
                                      bool drop_empty_grad = true) const {
    auto* var = this->SingleForwardInput(name);
    if (!var) {
      return nullptr;
    }
    auto var_name = var->Name();
    auto grad_var_name = framework::GradVarName(var_name);
    if (no_grad_set_.empty() || !no_grad_set_.count(grad_var_name)) {
      (*this->grad_to_var_)[grad_var_name] = var_name;
      VLOG(8) << "Valid gradients: " << grad_var_name;
    } else {
      // TODO(jiabin): Will this cause fill zeros error?
      grad_var_name = framework::kEmptyVarName;
      if (drop_empty_grad) return nullptr;
    }

    if (original_block_->HasVar(grad_var_name)) {
      // Copy Var from original block to active block, or create a new one.
      CopyVarFromOrig(grad_var_name);
      return StaticCompositeContext::Instance().GetBlock()->FindVar(
          grad_var_name);
    } else {
      return StaticCompositeContext::Instance().GetBlock()->Var(grad_var_name);
    }
  }

  framework::VarDesc* SingleOutputGrad(const std::string& name) const {
    auto* var = this->SingleForwardOutput(name);
    if (!var) {
      return nullptr;
    }
    auto var_name = var->Name();
    auto grad_var_name = framework::GradVarName(var_name);
    (*this->grad_to_var_)[grad_var_name] = var_name;
    VLOG(8) << "Valid gradients: " << grad_var_name;

    auto target_grad = StaticCompositeContext::Instance().GetTargetGradName();
    if (target_grad.find(grad_var_name) != target_grad.end()) {
      grad_var_name = target_grad.at(grad_var_name);
    }

    if (original_block_->HasVar(grad_var_name)) {
      // Copy Var from original block to active block, or create a new one.
      CopyVarFromOrig(grad_var_name);
      return StaticCompositeContext::Instance().GetBlock()->FindVar(
          grad_var_name);
    } else {
      return nullptr;
    }
  }

  std::vector<framework::VarDesc*> MultiInputGrad(
      const std::string& name) const {
    std::vector<std::string> ret_val;
    std::vector<framework::VarDesc*> input_grads;
    auto var_names = this->MultiForwardInputVarName(name);
    ret_val.reserve(var_names.size());
    std::transform(var_names.begin(),
                   var_names.end(),
                   std::back_inserter(ret_val),
                   [this](const std::string& fwd_var_name) -> std::string {
                     auto g_name = framework::GradVarName(fwd_var_name);
                     if (no_grad_set_.empty() || !no_grad_set_.count(g_name)) {
                       (*this->grad_to_var_)[g_name] = fwd_var_name;
                       return g_name;
                     } else {
                       return framework::kEmptyVarName;
                     }
                   });
    for (const auto& name : ret_val) {
      if (original_block_->HasVar(name)) {
        // Copy Var from original block to active block, or create a new one.
        CopyVarFromOrig(name);
        input_grads.emplace_back(
            StaticCompositeContext::Instance().GetBlock()->FindVar(name));
      } else {
        input_grads.emplace_back(
            StaticCompositeContext::Instance().GetBlock()->Var(name));
      }
    }
    return input_grads;
  }

  std::vector<framework::VarDesc*> MultiOutputGrad(
      const std::string& name) const {
    std::vector<std::string> ret_val;
    auto out_names = this->MultiForwardOutputVarName(name);
    ret_val.reserve(out_names.size());
    std::transform(out_names.begin(),
                   out_names.end(),
                   std::back_inserter(ret_val),
                   [this](const std::string& fwd_var_name) -> std::string {
                     auto g_name = framework::GradVarName(fwd_var_name);
                     (*this->grad_to_var_)[g_name] = fwd_var_name;
                     return g_name;
                   });
    std::vector<framework::VarDesc*> grad_out;
    for (auto name : ret_val) {
      auto target_grad = StaticCompositeContext::Instance().GetTargetGradName();
      if (target_grad.find(name) != target_grad.end()) {
        name = target_grad.at(name);
      }
      // TODO(jiabin): Will this cause fill zeros error?
      if (original_block_->HasVar(name)) {
        // Copy Var from original block to active block, or create a new one.
        CopyVarFromOrig(name);
        grad_out.emplace_back(
            StaticCompositeContext::Instance().GetBlock()->FindVar(name));
      } else {
        grad_out.emplace_back(
            StaticCompositeContext::Instance().GetBlock()->Var(name));
      }
    }
    return grad_out;
  }

  framework::VarDesc* SingleForwardInput(const std::string& name) const {
    // Copy Var from original block to active block, or create a new one.
    auto fwd_in_names = fwd_op_.Input(name);
    if (!fwd_in_names.empty()) {
      PADDLE_ENFORCE_EQ(
          fwd_in_names.size(),
          1,
          common::errors::InvalidArgument(
              "When calling SingleForward for op: %s's Input: %s, we should "
              "only get one input tensor, but we got %d instead.",
              fwd_op_.Type(),
              name,
              fwd_in_names.size()));
      CopyVarFromOrig(fwd_op_.Input(name).at(0));
      return StaticCompositeContext::Instance().GetBlock()->FindVar(
          fwd_op_.Input(name).at(0));
    } else {
      return nullptr;
    }
  }

  framework::VarDesc* SingleForwardOutput(const std::string& name) const {
    // Copy Var from original block to active block, or create a new one.
    auto fwd_out_names = fwd_op_.Output(name);
    if (!fwd_out_names.empty()) {
      PADDLE_ENFORCE_EQ(
          fwd_out_names.size(),
          1,
          common::errors::InvalidArgument(
              "When calling SingleForward for op: %s's Output: %s, we should "
              "only get one input tensor, but we got %d instead.",
              fwd_op_.Type(),
              name,
              fwd_out_names.size()));
      CopyVarFromOrig(fwd_op_.Output(name).at(0));
      return StaticCompositeContext::Instance().GetBlock()->FindVar(
          fwd_op_.Output(name).at(0));
    } else {
      return nullptr;
    }
  }

  std::vector<framework::VarDesc*> MultiForwardInput(
      const std::string& name) const {
    std::vector<framework::VarDesc*> result;
    for (const auto& n : fwd_op_.Input(name)) {
      // Copy Var from original block to active block, or create a new one.
      CopyVarFromOrig(n);
      result.emplace_back(
          StaticCompositeContext::Instance().GetBlock()->FindVar(n));
    }
    return result;
  }

  std::vector<framework::VarDesc*> MultiForwardOutput(
      const std::string& name) const {
    std::vector<framework::VarDesc*> result;
    for (const auto& n : fwd_op_.Output(name)) {
      // Copy Var from original block to active block, or create a new one.
      CopyVarFromOrig(n);
      result.emplace_back(
          StaticCompositeContext::Instance().GetBlock()->FindVar(n));
    }
    return result;
  }

  void RecoverOutputName(const Tensor& output, const std::string& origin_name) {
    if (origin_name == framework::kEmptyVarName) return;
    VLOG(4) << "Recover: "
            << static_cast<DescTensor*>(output.impl().get())->Name()
            << " To: " << origin_name;
    prim::StaticCompositeContext::Instance().GetBlock()->RenameVar(
        static_cast<DescTensor*>(output.impl().get())->Name(), origin_name);
  }

  void RecoverOutputName(const std::vector<Tensor>& outputs,
                         const std::vector<std::string>& origin_names) {
    PADDLE_ENFORCE_EQ(outputs.size(),
                      origin_names.size(),
                      common::errors::InvalidArgument(
                          "The size of outputs must be equal to the size "
                          "of the origin_names, but got outputs size: %d, "
                          "origin_names size: %d.",
                          outputs.size(),
                          origin_names.size()));
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (origin_names[i] == framework::kEmptyVarName) continue;
      prim::StaticCompositeContext::Instance().GetBlock()->RenameVar(
          static_cast<DescTensor*>(outputs[i].impl().get())->Name(),
          origin_names[i]);
    }
  }

  std::vector<std::string> MultiForwardOutputVarName(
      const std::string& name) const {
    return fwd_op_.Output(name);
  }

  std::vector<std::string> MultiForwardInputVarName(
      const std::string& name) const {
    return fwd_op_.Input(name);
  }

  static std::vector<std::string> EmptyInput() { return {}; }

  static std::vector<std::string> EmptyOutput() { return {}; }

  static std::vector<std::string> EmptyInputGrad() { return {}; }

  static std::vector<std::string> EmptyOutputGrad() { return {}; }

  std::vector<std::string> InputNames() const {
    return this->fwd_op_.InputNames();
  }

  std::vector<std::string> OutputNames() const {
    return this->fwd_op_.OutputNames();
  }

  const std::unordered_map<std::string, framework::Attribute>& Attrs() const {
    return fwd_op_.GetAttrMap();
  }

  const std::unordered_map<std::string, framework::Attribute>& RuntimeAttrs()
      const {
    LOG(WARNING) << "CompositeGradOpMaker doesn't support use runtime attrs, "
                    "but find the op"
                 << fwd_op_.Type() << "use runtime attr.";
    return fwd_op_.GetRuntimeAttrMap();
  }

  const framework::Attribute& GetAttr(const std::string& name) const {
    auto& map = fwd_op_.GetAttrMap();
    auto it = map.find(name);
    PADDLE_ENFORCE_NE(
        it,
        map.end(),
        common::errors::NotFound("Cannot find attribute (%s).", name));
    return it->second;
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return PADDLE_GET_CONST(T, GetAttr(name));
  }

  std::string ForwardOpType() const { return this->fwd_op_.Type(); }
  const framework::BlockDesc* GetForwardOpBlock() const {
    return fwd_op_.Block();
  }

 protected:
  bool HasInput(const std::string& name) const {
    return (fwd_op_.Inputs().count(name) > 0);
  }

  bool HasOutput(const std::string& name) const {
    return (fwd_op_.Outputs().count(name) > 0);
  }

 private:
  const framework::OpDesc& fwd_op_;
  const std::unordered_set<std::string>& no_grad_set_;
  std::unordered_map<std::string, std::string>* grad_to_var_;
  const framework::BlockDesc* original_block_;
  framework::ProgramDesc acting_program_;

 protected:
  std::vector<framework::BlockDesc*> grad_block_;
};

}  // namespace prim
}  // namespace paddle
