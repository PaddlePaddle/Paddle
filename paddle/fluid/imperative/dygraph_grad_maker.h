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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {

class GradOpBaseMakerBase {
 public:
  explicit GradOpBaseMakerBase(const OpBase* fw_op_base,
                               const NameVarBaseMap& var_base_map_in,
                               const NameVarBaseMap& var_base_map_out)
      : fw_op_base_(fw_op_base),
        var_base_map_in_(var_base_map_in),
        var_base_map_out_(var_base_map_out) {}

  virtual ~GradOpBaseMakerBase() = default;
  virtual std::vector<std::unique_ptr<OpBase>> operator()() const = 0;

  std::vector<std::shared_ptr<VarBase>> InputGrad(
      const std::string& name, bool drop_empty_grad = true) const {
    return GetVarBaseList(name, true, true);
  }

  std::vector<std::shared_ptr<VarBase>> OutputGrad(
      const std::string& name) const {
    return GetVarBaseList(name, true, false);
  }

  std::vector<std::shared_ptr<VarBase>> Input(const std::string name) const {
    return GetVarBaseList(name, false, true);
  }

  std::vector<std::shared_ptr<VarBase>> Output(const std::string& name) const {
    return GetVarBaseList(name, false, false);
  }

  std::vector<std::shared_ptr<VarBase>> Empty() const { return {}; }

  std::vector<std::string> InputNames() const {
    std::vector<std::string> vec_temp;
    vec_temp.reserve(var_base_map_in_.size());
    for (auto& it : var_base_map_in_) {
      vec_temp.emplace_back(it.first);
    }

    return vec_temp;
  }

  std::vector<std::string> OutputNames() const {
    std::vector<std::string> vec_temp;
    vec_temp.reserve(var_base_map_out_.size());
    for (auto& it : var_base_map_out_) {
      vec_temp.emplace_back(it.first);
    }

    return vec_temp;
  }

  const std::unordered_map<std::string, framework::Attribute>& Attrs() const {
    return fw_op_base_->Attrs();
  }

  const framework::Attribute& GetAttr(const std::string& name) const {
    auto& map = fw_op_base_->Attrs();
    auto it = map.find(name);
    PADDLE_ENFORCE(it != map.end(),
                   "Cannot find attribute [%s] in operator [%s]", name,
                   fw_op_base_->Type());

    return it->second;
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return boost::get<T>(GetAttr(name));
  }

  std::string ForwardOpType() const { return fw_op_base_->Type(); }

 protected:
  bool HasInput(const std::string& name) const {
    auto it = var_base_map_in_.find(name);

    return it != var_base_map_in_.end();
  }

  bool HasOutput(const std::string name) const {
    auto it = var_base_map_out_.find(name);

    return it != var_base_map_out_.end();
  }

 private:
  std::vector<std::shared_ptr<VarBase>> GetVarBaseList(const std::string& name,
                                                       bool is_grad,
                                                       bool is_input) const {
    const NameVarBaseMap& data_map =
        is_input ? var_base_map_in_ : var_base_map_out_;
    auto iterator = data_map.find(name);

    std::vector<std::shared_ptr<imperative::VarBase>> vec_temp;
    if (iterator != data_map.end()) {
      vec_temp.reserve(iterator->second.size());

      for (auto& var_base_temp : iterator->second) {
        if (is_grad) {
          if (!var_base_temp->HasGradVar()) {
            VLOG(6) << "GradVarBase of var " << var_base_temp->Name()
                    << " in OP " << fw_op_base_->Type() << " is null";
            var_base_temp->MutableGradVarBase();
          }
          auto grad_var_base_tmp = var_base_temp->GradVarBase();
          if (!is_input) {
            auto* tensor = grad_var_base_tmp->MutableVar()
                               ->GetMutable<framework::LoDTensor>();
            tensor->Resize(
                var_base_temp->Var().Get<framework::LoDTensor>().dims());
          }
          vec_temp.emplace_back(grad_var_base_tmp);
        } else {
          vec_temp.emplace_back(var_base_temp);
        }
      }
    }

    return vec_temp;
  }

 private:
  const OpBase* fw_op_base_;
  const NameVarBaseMap& var_base_map_in_;
  const NameVarBaseMap& var_base_map_out_;

 protected:
  std::vector<framework::BlockDesc*> grad_block_;
};

}  // namespace imperative
}  // namespace paddle
