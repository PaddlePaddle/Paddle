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

#pragma once

#include "paddle/framework/ddim.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/shape_inference.h"

namespace paddle {
namespace framework {

class BlockDesc {
 public:
  BlockDesc(const std::map<std::string, VarDesc*>& var_descs)
      : var_descs_(var_descs) {}
  ~BlockDesc() {}

  VarDesc* get_var(const std::string& name) const {
    PADDLE_ENFORCE(var_descs_.count(name) == 1, "%s must be in Block", name);
    return var_descs_.at(name);
  }

 private:
  const std::map<std::string, VarDesc*>& var_descs_;
};

class CompileTimeInferShapeContext : public InferShapeContextBase {
 public:
  CompileTimeInferShapeContext(std::unique_ptr<OperatorBase>& op,
                               const BlockDesc& block_desc)
      : op_(std::move(op)), block_desc_(block_desc) {}

  DDim get_input_dim(const std::string& name) const {
    return get_dim(op_->Input(name));
  }

  void set_input_dim(const std::string& name, const DDim& dim) const {
    set_dim(op_->Input(name), dim);
  }

  DDim get_output_dim(const std::string& name) const {
    return get_dim(op_->Output(name));
  }

  void set_output_dim(const std::string& name, const DDim& dim) const {
    set_dim(op_->Output(name), dim);
  }

  AttrReader attrs() const { return AttrReader(op_->Attrs()); }

 private:
  DDim get_dim(const std::string& name) const {
    VarDesc* desc = block_desc_.get_var(name);
    std::vector<int64_t> dim;
    int length = desc->lod_tensor().dims().size();
    dim.reserve(length);
    std::copy(desc->lod_tensor().dims().begin(),
              desc->lod_tensor().dims().end(), std::back_inserter(dim));
    return make_ddim(dim);
  }

  void set_dim(const std::string& name, const DDim& dim) const {
    VarDesc* desc = block_desc_.get_var(name);
    auto tensor = desc->mutable_lod_tensor();
    tensor->clear_dims();
    for (int i = 0; i < dim.size(); ++i) {
      tensor->add_dims(static_cast<int>(dim[i]));
    }
  }

  std::unique_ptr<OperatorBase> op_;
  const BlockDesc& block_desc_;
};

class RunTimeInferShapeContext : public InferShapeContextBase {
 public:
  RunTimeInferShapeContext(const OperatorBase& op, const Scope& scope)
      : op_(op), scope_(scope) {}

  DDim get_input_dim(const std::string& name) const {
    return get_dim(op_.Input(name));
  }

  void set_input_dim(const std::string& name, const DDim& dim) const {
    set_dim(op_.Input(name), dim);
  }

  DDim get_output_dim(const std::string& name) const {
    return get_dim(op_.Output(name));
  }

  void set_output_dim(const std::string& name, const DDim& dim) const {
    set_dim(op_.Output(name), dim);
  }

  AttrReader attrs() const { return AttrReader(op_.Attrs()); }

 private:
  DDim get_dim(const std::string& name) const {
    Tensor* t = scope_.FindVar(op_.Input(name))->GetMutable<Tensor>();
    return t->dims();
  }

  void set_dim(const std::string& name, const DDim& dim) const {
    Tensor* t = scope_.FindVar(name)->GetMutable<Tensor>();
    t->Resize(dim);
  }

  const OperatorBase& op_;
  const Scope& scope_;
};

}  // namespace framework
}  // namespace paddle
