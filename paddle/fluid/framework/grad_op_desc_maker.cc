// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace framework {

GradOpDescMakerBase::GradOpDescMakerBase(
    const OpDesc& fwd_op, const std::unordered_set<std::string>& no_grad_set,
    GradToVarMapType* grad_to_var, const std::vector<BlockDesc*>& grad_block)
    : fwd_op_(fwd_op),
      no_grad_set_(no_grad_set),
      grad_to_var_(grad_to_var),
      grad_block_(grad_block) {
  LOG(INFO) << "Ctor " << fwd_op_.Type() << " addr " << grad_to_var_;
}

GradOpDescMakerBase::~GradOpDescMakerBase() {}

std::vector<std::string> GradOpDescMakerBase::InputGrad(
    const std::string& name, bool drop_empty_grad) const {
  std::vector<std::string> ret_val;
  auto var_names = this->Input(name);
  ret_val.reserve(var_names.size());
  std::transform(
      var_names.begin(), var_names.end(), std::back_inserter(ret_val),
      [this](const std::string& fwd_var_name) -> std::string {
        auto g_name = GradVarName(fwd_var_name);
        if (no_grad_set_.empty() || !no_grad_set_.count(g_name)) {
          LOG(INFO) << "Offset = "
                    << reinterpret_cast<uintptr_t>(&(this->grad_to_var_)) -
                           reinterpret_cast<uintptr_t>(this);
          LOG(INFO) << "Sizeof = " << sizeof(this->grad_to_var_);
          LOG(INFO) << "S " << ForwardOpType() << " " << fwd_var_name << " "
                    << g_name << " " << this->grad_to_var_->size();
          (*this->grad_to_var_)[g_name] = fwd_var_name;
          LOG(INFO) << "E " << ForwardOpType() << " " << fwd_var_name << " "
                    << g_name << " " << this->grad_to_var_->size();
          return g_name;
        } else {
          return kEmptyVarName;
        }
      });
  if (!drop_empty_grad) {
    return ret_val;
  }
  PADDLE_ENFORCE_LE(var_names.size(), 1UL,
                    "BUG from operator developer:"
                    " for input argument with a list of variables, "
                    " drop_empty_grad is not allowed because it makes"
                    " the correspondence bewteen a variable and its gradient"
                    " ambiguous."
                    " Op type %s",
                    fwd_op_.Type());

  std::vector<std::string> dropped_ret_val;
  dropped_ret_val.reserve(ret_val.size());
  std::copy_if(ret_val.begin(), ret_val.end(),
               std::back_inserter(dropped_ret_val),
               [](const std::string& str) { return str != kEmptyVarName; });
  return dropped_ret_val;
}

std::vector<std::string> GradOpDescMakerBase::OutputGrad(
    const std::string& name) const {
  std::vector<std::string> ret_val;
  auto onames = this->Output(name);
  ret_val.reserve(onames.size());
  std::transform(
      onames.begin(), onames.end(), std::back_inserter(ret_val),
      [this](const std::string& fwd_var_name) -> std::string {
        auto g_name = GradVarName(fwd_var_name);
        LOG(INFO) << "Offset = "
                  << reinterpret_cast<uintptr_t>(&(this->grad_to_var_)) -
                         reinterpret_cast<uintptr_t>(this);
        LOG(INFO) << "Sizeof = " << sizeof(this->grad_to_var_);
        LOG(INFO) << "SS " << ForwardOpType() << " " << fwd_var_name << " "
                  << g_name << " " << this->grad_to_var_->size();
        (*this->grad_to_var_)[g_name] = fwd_var_name;
        LOG(INFO) << "EE " << ForwardOpType() << " " << fwd_var_name << " "
                  << g_name << " " << this->grad_to_var_->size();
        return g_name;
      });
  return ret_val;
}

std::vector<std::string> GradOpDescMakerBase::Empty() const { return {}; }

std::vector<std::string> GradOpDescMakerBase::InputNames() const {
  return this->fwd_op_.InputNames();
}

std::vector<std::string> GradOpDescMakerBase::OutputNames() const {
  return this->fwd_op_.OutputNames();
}

std::vector<std::string> GradOpDescMakerBase::Input(
    const std::string& name) const {
  return fwd_op_.Input(name);
}

std::vector<std::string> GradOpDescMakerBase::Output(
    const std::string& name) const {
  return fwd_op_.Output(name);
}

const AttributeMap& GradOpDescMakerBase::Attrs() const {
  return fwd_op_.GetAttrMap();
}

const Attribute& GradOpDescMakerBase::GetAttr(const std::string& name) const {
  auto& map = fwd_op_.GetAttrMap();
  auto it = map.find(name);
  PADDLE_ENFORCE(it != map.end(), "Cannot find attribute %s", name);
  return it->second;
}

std::string GradOpDescMakerBase::ForwardOpType() const {
  return this->fwd_op_.Type();
}

bool GradOpDescMakerBase::HasInput(const std::string& name) const {
  return (fwd_op_.Inputs().count(name) > 0);
}

void* AnyFunc(void* obj) {
  auto* map = reinterpret_cast<GradToVarMapType*>(obj);
  (*map)["_generated_var_0"] = "_generated_var_0@GRAD";
  return map;
}

OpDesc* NewOpDesc() { return new OpDesc(); }

std::vector<std::unique_ptr<OpDesc>> GradOpMakerFN::operator()(
    const OpDesc& op_desc, const std::unordered_set<std::string>& no_grad_set,
    GradToVarMapType* grad_to_var,
    const std::vector<BlockDesc*>& grad_blocks) const {
  if (func_) {
    return func_(op_desc, no_grad_set, grad_to_var, grad_blocks);
  } else {
    return reserved_func_(op_desc, no_grad_set, grad_to_var, grad_blocks,
                          reserved_);
  }
}

}  // namespace framework
}  // namespace paddle
