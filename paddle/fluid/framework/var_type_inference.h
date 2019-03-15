/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include <vector>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace framework {

class OpDesc;
class BlockDesc;
// default infer var type context
class InferVarTypeContext {
 public:
  InferVarTypeContext(const OpDesc* op, BlockDesc* block)
      : op_(op), block_(block) {}

  Attribute GetAttr(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(op_);
    return op_->GetAttr(name);
  }

  inline bool HasVar(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_);
    return block_->FindVarRecursive(name) != nullptr;
  }

  inline bool HasInput(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(op_);
    return op_->Inputs().count(name) > 0;
  }

  inline bool HasOutput(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(op_);
    return op_->Outputs().count(name) > 0;
  }

  inline const std::vector<std::string>& Input(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(op_);
    return op_->Input(name);
  }

  inline const std::vector<std::string>& Output(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(op_);
    return op_->Output(name);
  }

  inline proto::VarType::Type GetType(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_);
    return block_->FindRecursiveOrCreateVar(name).GetType();
  }

  inline void SetType(const std::string& name, proto::VarType::Type type) {
    PADDLE_ENFORCE_NOT_NULL(block_);
    block_->FindRecursiveOrCreateVar(name).SetType(type);
  }

  inline proto::VarType::Type GetDataType(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_);
    return block_->FindRecursiveOrCreateVar(name).GetDataType();
  }

  inline void SetDataType(const std::string& name, proto::VarType::Type type) {
    PADDLE_ENFORCE_NOT_NULL(block_);
    block_->FindRecursiveOrCreateVar(name).SetDataType(type);
  }

  inline std::vector<int64_t> GetShape(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_);
    return block_->FindRecursiveOrCreateVar(name).GetShape();
  }

  inline void SetShape(const std::string& name,
                       const std::vector<int64_t>& dims) {
    PADDLE_ENFORCE_NOT_NULL(block_);
    block_->FindRecursiveOrCreateVar(name).SetShape(dims);
  }

  inline int32_t GetLoDLevel(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_);
    return block_->FindRecursiveOrCreateVar(name).GetLoDLevel();
  }

  inline void SetLoDLevel(const std::string& name, int32_t lod_level) {
    PADDLE_ENFORCE_NOT_NULL(block_);
    block_->FindRecursiveOrCreateVar(name).SetLoDLevel(lod_level);
  }

 private:
  const OpDesc* op_;
  BlockDesc* block_;
};

// infer var type context for imperative mode
class RuntimeInferVarTypeContext : public InferVarTypeContext {
 public:
  RuntimeInferVarTypeContext() : InferVarTypeContext(nullptr, nullptr) {}
};

class VarTypeInference {
 public:
  virtual ~VarTypeInference() {}
  virtual void operator()(InferVarTypeContext& context) const = 0;  // NOLINT
};

class PassInDtypeAndVarTypeToOutput : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext& ctx) const final {  // NOLINT
    auto in_out_var_names = this->GetInputOutputWithSameType();

    for (auto& i_o_n : in_out_var_names) {
      auto& x_name = ctx.Input(i_o_n.first).at(0);
      auto& out_name = ctx.Output(i_o_n.second).at(0);

      ctx.SetType(out_name, ctx.GetType(x_name));
      ctx.SetDataType(out_name, ctx.GetDataType(x_name));
    }
  }

 protected:
  virtual std::unordered_map<std::string, std::string>
  GetInputOutputWithSameType() const = 0;
};

}  // namespace framework
}  // namespace paddle
