// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/fluid/framework/variable.h"

namespace egr {
class EagerTensor;
}  // namespace egr

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework

namespace imperative {

class VarBase;
class VariableWrapper;

void InitializeVariable(paddle::framework::Variable* var,
                        paddle::framework::proto::VarType::Type var_type);
paddle::framework::proto::VarType::Type GetDtypeFromVar(
    const paddle::framework::Variable& var);
const paddle::platform::Place& GetPlaceFromVar(
    const paddle::framework::Variable& var);
void CopyVariable(const paddle::framework::Variable& src_var,
                  paddle::framework::Variable* dst_var);
std::string GetNameFromVar(const egr::EagerTensor& tensor);
std::string GetNameFromVar(const VarBase& var);
bool CheckCachedKey(std::shared_ptr<egr::EagerTensor> tensor,
                    const paddle::framework::OpKernelType& key);
bool CheckCachedKey(std::shared_ptr<VarBase> var,
                    const paddle::framework::OpKernelType& key);
void SetCachedValue(std::shared_ptr<egr::EagerTensor> tensor,
                    const paddle::framework::OpKernelType& key,
                    std::shared_ptr<VariableWrapper> res);
void SetCachedValue(std::shared_ptr<VarBase> var,
                    const paddle::framework::OpKernelType& key,
                    std::shared_ptr<VariableWrapper> res);
std::shared_ptr<VariableWrapper> GetCachedValue(
    std::shared_ptr<egr::EagerTensor> tensor,
    const paddle::framework::OpKernelType& key);
std::shared_ptr<VariableWrapper> GetCachedValue(
    std::shared_ptr<VarBase> var, const paddle::framework::OpKernelType& key);
void SetType(std::shared_ptr<VarBase> var,
             framework::proto::VarType::Type type);

void SetType(std::shared_ptr<egr::EagerTensor> var,
             framework::proto::VarType::Type type);

framework::proto::VarType::Type GetType(std::shared_ptr<egr::EagerTensor> var);

framework::proto::VarType::Type GetType(std::shared_ptr<VarBase> var);

framework::proto::VarType::Type GetDataType(
    std::shared_ptr<egr::EagerTensor> var);

framework::proto::VarType::Type GetDataType(std::shared_ptr<VarBase> var);

const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<paddle::imperative::VarBase>& var);
const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<VariableWrapper>& var);
}  // namespace imperative
}  // namespace paddle
