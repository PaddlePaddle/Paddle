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
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace framework {

class OpDesc;
class BlockDesc;
// default infer var type context

static const int ALL_ELEMENTS = -1;

class InferVarTypeContext {
 public:
  InferVarTypeContext(const OpDesc* op, BlockDesc* block)
      : op_(op), block_(block) {}

  virtual ~InferVarTypeContext() {}

  virtual Attribute GetAttr(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(op_);
    return op_->GetAttr(name);
  }

  virtual bool HasVar(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_);
    return block_->FindVarRecursive(name) != nullptr;
  }

  virtual bool HasInput(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(op_);
    auto& inputs = op_->Inputs();
    auto input = inputs.find(name);
    return input != inputs.end() && !input->second.empty();
  }

  virtual bool HasOutput(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(op_);
    auto& outputs = op_->Outputs();
    auto output = outputs.find(name);
    return output != outputs.end() && !output->second.empty();
  }

  virtual bool InputTypeAnyOf(const std::string& name,
                              proto::VarType::Type type) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& inputs = op_->Input(name);
    return std::any_of(inputs.begin(), inputs.end(),
                       [this, &type](const std::string& name) {
                         return this->GetType(name) == type;
                       });
  }

  virtual bool InputTypeAllOf(const std::string& name,
                              proto::VarType::Type type) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& inputs = op_->Input(name);
    return std::all_of(inputs.begin(), inputs.end(),
                       [this, &type](const std::string& name) {
                         return this->GetType(name) == type;
                       });
  }

  // not available in dygraph mode
  virtual const std::vector<std::string>& Input(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return op_->Input(name);
  }

  // not available in dygraph mode
  virtual const std::vector<std::string>& Output(
      const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return op_->Output(name);
  }

  virtual void SyncTypeAndDataType(const std::string& input_name,
                                   const std::string& output_name,
                                   int index = 0) {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& x_name = op_->Input(input_name).at(index);
    auto& out_name = op_->Output(output_name).at(index);

    if (x_name != out_name) {
      this->SetType(out_name, this->GetType(x_name));
      this->SetDataType(out_name, this->GetDataType(x_name));
    }
  }

  virtual void SetOutputType(const std::string& name, proto::VarType::Type type,
                             int index = 0) {
    if (ALL_ELEMENTS == index) {
      for (const auto& var_name : op_->Output(name)) {
        this->SetType(var_name, type);
      }
    } else {
      auto& var_name = op_->Output(name).at(index);
      this->SetType(var_name, type);
    }
  }

  // used in paddle/fluid/operators/save_op.cc
  // legacy function, maybe removed in the future, don't use in new code
  // use SetOutputType instead
  virtual void SetType(const std::string& name, proto::VarType::Type type) {
    PADDLE_ENFORCE_NOT_NULL(
        block_, platform::errors::PreconditionNotMet("op_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetType(type);
  }

  virtual proto::VarType::Type GetInputType(const std::string& name,
                                            const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return this->GetType(op_->Input(name).at(index));
  }

  // not available in dygraph mode
  virtual proto::VarType::Type GetType(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetType();
  }

  virtual proto::VarType::Type GetInputDataType(const std::string& name,
                                                const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return this->GetDataType(op_->Input(name).at(index));
  }

  virtual proto::VarType::Type GetOutputDataType(const std::string& name,
                                                 const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return this->GetDataType(op_->Output(name).at(index));
  }

  // not available in dygraph mode
  virtual proto::VarType::Type GetDataType(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetDataType();
  }

  virtual void SetOutputDataType(const std::string& name,
                                 proto::VarType::Type type, int index = 0) {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    if (ALL_ELEMENTS == index) {
      for (const auto& var_name : op_->Output(name)) {
        this->SetDataType(var_name, type);
      }
    } else {
      auto& var_name = op_->Output(name).at(index);
      this->SetDataType(var_name, type);
    }
  }

  // not available in dygraph mode
  virtual void SetDataType(const std::string& name, proto::VarType::Type type) {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetDataType(type);
  }

  virtual std::vector<proto::VarType::Type> GetDataTypes(
      const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetDataTypes();
  }

  virtual void SetDataTypes(
      const std::string& name,
      const std::vector<proto::VarType::Type>& multiple_data_type) {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetDataTypes(multiple_data_type);
  }

  virtual std::vector<int64_t> GetShape(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetShape();
  }

  virtual void SetShape(const std::string& name,
                        const std::vector<int64_t>& dims) {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetShape(dims);
  }

  virtual int32_t GetLoDLevel(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetLoDLevel();
  }

  virtual void SetLoDLevel(const std::string& name, int32_t lod_level) {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetLoDLevel(lod_level);
  }

  virtual bool IsDygraph() const { return false; }

 protected:
  const OpDesc* op_;
  BlockDesc* block_;
};

class VarTypeInference {
 public:
  virtual ~VarTypeInference() {}
  virtual void operator()(InferVarTypeContext* context) const = 0;  // NOLINT
};

class PassInDtypeAndVarTypeToOutput : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const final {  // NOLINT
    auto& in_out_var_names = this->GetInputOutputWithSameType();

    for (auto& i_o_n : in_out_var_names) {
      ctx->SyncTypeAndDataType(i_o_n.first, i_o_n.second);
    }
  }

 protected:
  virtual std::unordered_map<std::string, std::string>&
  GetInputOutputWithSameType() const = 0;
};

}  // namespace framework
}  // namespace paddle
