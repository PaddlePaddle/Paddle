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

class BlockDesc;
class OpDesc;
class StaticGraphVarTypeInference;
// default infer var type context

static const int ALL_ELEMENTS = -1;

class InferVarTypeContext {
  friend class StaticGraphVarTypeInference;

 public:
  InferVarTypeContext(const OpDesc* op, BlockDesc* block)
      : op_(op), block_(block) {}

  virtual ~InferVarTypeContext() {}

  virtual Attribute GetAttr(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return op_->GetAttr(name);
  }

  virtual bool HasInput(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& inputs = op_->Inputs();
    auto input = inputs.find(name);
    return input != inputs.end() && !input->second.empty();
  }

  virtual bool HasOutput(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& outputs = op_->Outputs();
    auto output = outputs.find(name);
    return output != outputs.end() && !output->second.empty();
  }

  virtual size_t InputSize(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return op_->Inputs().at(name).size();
  }

  virtual const std::string& InputVarName(const std::string& name,
                                          const int index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return op_->Inputs().at(name)[index];
  }

  virtual bool InputTypeAnyOf(const std::string& name,
                              proto::VarType::Type type) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& inputs = op_->Input(name);
    return std::any_of(inputs.begin(), inputs.end(),
                       [this, &type](const std::string& name) {
                         return this->GetVarType(name) == type;
                       });
  }

  virtual bool InputTypeAllOf(const std::string& name,
                              proto::VarType::Type type) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& inputs = op_->Input(name);
    return std::all_of(inputs.begin(), inputs.end(),
                       [this, &type](const std::string& name) {
                         return this->GetVarType(name) == type;
                       });
  }

  virtual void SyncTypeAndDataType(const std::string& input_name,
                                   const std::string& output_name,
                                   int index = 0) {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& x_name = op_->Input(input_name).at(index);
    auto& out_name = op_->Output(output_name).at(index);

    if (x_name != out_name) {
      this->SetVarType(out_name, this->GetVarType(x_name));
      this->SetVarDataType(out_name, this->GetVarDataType(x_name));
    }
  }

  virtual void SetOutputType(const std::string& name, proto::VarType::Type type,
                             int index = 0) {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    if (ALL_ELEMENTS == index) {
      for (const auto& var_name : op_->Output(name)) {
        this->SetVarType(var_name, type);
      }
    } else {
      auto& var_name = op_->Output(name).at(index);
      this->SetVarType(var_name, type);
    }
  }

  virtual proto::VarType::Type GetInputType(const std::string& name,
                                            const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return this->GetVarType(op_->Input(name).at(index));
  }

  virtual proto::VarType::Type GetOutputType(const std::string& name,
                                             const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return this->GetVarType(op_->Output(name).at(index));
  }

  virtual proto::VarType::Type GetInputDataType(const std::string& name,
                                                const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return this->GetVarDataType(op_->Input(name).at(index));
  }

  virtual void SetOutputDataType(const std::string& name,
                                 proto::VarType::Type type, int index = 0) {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    if (ALL_ELEMENTS == index) {
      for (const auto& var_name : op_->Output(name)) {
        this->SetVarDataType(var_name, type);
      }
    } else {
      auto& var_name = op_->Output(name).at(index);
      this->SetVarDataType(var_name, type);
    }
  }

  virtual std::vector<proto::VarType::Type> GetInputDataTypes(
      const std::string& name, const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return this->GetVarDataTypes(op_->Input(name).at(index));
  }

  virtual void SetOutputDataTypes(
      const std::string& name,
      const std::vector<proto::VarType::Type>& multiple_data_type,
      const int& index = 0) {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& var_name = op_->Output(name).at(index);
    this->SetVarDataTypes(var_name, multiple_data_type);
  }

  virtual std::vector<int64_t> GetInputShape(const std::string& name,
                                             const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& var_name = op_->Input(name).at(index);
    return this->GetVarShape(var_name);
  }

  virtual void SetOutputShape(const std::string& name,
                              const std::vector<int64_t>& dims,
                              const int& index = 0) {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& var_name = op_->Output(name).at(index);
    this->SetVarShape(var_name, dims);
  }

  virtual int32_t GetInputLoDLevel(const std::string& name,
                                   const int& index = 0) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& var_name = op_->Input(name).at(index);
    return this->GetVarLoDLevel(var_name);
  }

  virtual void SetOutputLoDLevel(const std::string& name, int32_t lod_level,
                                 const int& index = 0) {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    auto& var_name = op_->Output(name).at(index);
    this->SetVarLoDLevel(var_name, lod_level);
  }

  // add a speical API for save_op
  // avoid use this API for common logic
  virtual void InsertVar(const std::string& var_name,
                         proto::VarType::Type var_type) {
    if (!IsDygraph()) this->SetVarType(var_name, var_type);
  }

  virtual bool IsDygraph() const { return false; }

 protected:
  virtual bool HasVar(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindVarRecursive(name) != nullptr;
  }

  virtual const std::vector<std::string>& InputVars(
      const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return op_->Input(name);
  }

  virtual const std::vector<std::string>& OutputVars(
      const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(
        op_, platform::errors::PreconditionNotMet("op_ should not be null"));
    return op_->Output(name);
  }

  virtual proto::VarType::Type GetVarType(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetType();
  }

  virtual void SetVarType(const std::string& name, proto::VarType::Type type) {
    PADDLE_ENFORCE_NOT_NULL(
        block_, platform::errors::PreconditionNotMet("op_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetType(type);
  }

  virtual proto::VarType::Type GetVarDataType(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetDataType();
  }

  virtual void SetVarDataType(const std::string& name,
                              proto::VarType::Type type) {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetDataType(type);
  }

  virtual std::vector<proto::VarType::Type> GetVarDataTypes(
      const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetDataTypes();
  }

  virtual void SetVarDataTypes(
      const std::string& name,
      const std::vector<proto::VarType::Type>& multiple_data_type) {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetDataTypes(multiple_data_type);
  }

  virtual std::vector<int64_t> GetVarShape(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetShape();
  }

  virtual void SetVarShape(const std::string& name,
                           const std::vector<int64_t>& dims) {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetShape(dims);
  }

  virtual int32_t GetVarLoDLevel(const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    return block_->FindRecursiveOrCreateVar(name).GetLoDLevel();
  }

  virtual void SetVarLoDLevel(const std::string& name, int32_t lod_level) {
    PADDLE_ENFORCE_NOT_NULL(block_, platform::errors::PreconditionNotMet(
                                        "block_ should not be null"));
    block_->FindRecursiveOrCreateVar(name).SetLoDLevel(lod_level);
  }

 protected:
  const OpDesc* op_;
  BlockDesc* block_;
};

class VarTypeInference {
 public:
  virtual ~VarTypeInference() {}
  virtual void operator()(InferVarTypeContext* context) const = 0;  // NOLINT
};

class StaticGraphVarTypeInference : public VarTypeInference {
 protected:
  bool HasVar(InferVarTypeContext* ctx, const std::string& name) const {
    return ctx->HasVar(name);
  }

  const std::vector<std::string>& Input(InferVarTypeContext* ctx,
                                        const std::string& name) const {
    return ctx->InputVars(name);
  }

  const std::vector<std::string>& Output(InferVarTypeContext* ctx,
                                         const std::string& name) const {
    return ctx->OutputVars(name);
  }

  proto::VarType::Type GetType(InferVarTypeContext* ctx,
                               const std::string& name) const {
    return ctx->GetVarType(name);
  }

  void SetType(InferVarTypeContext* ctx, const std::string& name,
               proto::VarType::Type type) const {
    ctx->SetVarType(name, type);
  }

  proto::VarType::Type GetDataType(InferVarTypeContext* ctx,
                                   const std::string& name) const {
    return ctx->GetVarDataType(name);
  }

  void SetDataType(InferVarTypeContext* ctx, const std::string& name,
                   proto::VarType::Type type) const {
    ctx->SetVarDataType(name, type);
  }

  std::vector<proto::VarType::Type> GetDataTypes(
      InferVarTypeContext* ctx, const std::string& name) const {
    return ctx->GetVarDataTypes(name);
  }

  void SetDataTypes(
      InferVarTypeContext* ctx, const std::string& name,
      const std::vector<proto::VarType::Type>& multiple_data_type) {
    return ctx->SetVarDataTypes(name, multiple_data_type);
  }

  std::vector<int64_t> GetShape(InferVarTypeContext* ctx,
                                const std::string& name) const {
    return ctx->GetVarShape(name);
  }

  void SetShape(InferVarTypeContext* ctx, const std::string& name,
                const std::vector<int64_t>& dims) const {
    ctx->SetVarShape(name, dims);
  }

  int32_t GetLoDLevel(InferVarTypeContext* ctx, const std::string& name) const {
    return ctx->GetVarLoDLevel(name);
  }

  void SetLoDLevel(InferVarTypeContext* ctx, const std::string& name,
                   int32_t lod_level) const {
    ctx->SetVarLoDLevel(name, lod_level);
  }
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
