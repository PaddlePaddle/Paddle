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
class EagerVariable;
}  // namespace egr
namespace phi {
class DenseTensor;
}
namespace paddle {
namespace framework {
class Variable;
class OpKernelType;
}  // namespace framework

namespace imperative {

class VarBase;
class VariableWrapper;

void InitializeVariable(paddle::framework::Variable* var,
                        paddle::framework::proto::VarType::Type var_type);
template <typename VarType>
const paddle::platform::Place& GetPlace(const std::shared_ptr<VarType>& var);
template <typename VarType>
const std::string& GetNameFromVar(std::shared_ptr<VarType> var);

template <typename VarType>
bool CheckCachedKey(std::shared_ptr<VarType> tensor,
                    const paddle::framework::OpKernelType& key);
template <typename VarType>
void SetCachedValue(std::shared_ptr<VarType> tensor,
                    const paddle::framework::OpKernelType& key,
                    std::shared_ptr<VarType> res);
template <typename VarType>
std::shared_ptr<VariableWrapper> GetCachedValue(
    std::shared_ptr<VarType> tensor,
    const paddle::framework::OpKernelType& key);

template <typename VarType>
void SetType(std::shared_ptr<VarType> var,
             framework::proto::VarType::Type type);

template <typename VarType>
framework::proto::VarType::Type GetType(std::shared_ptr<VarType> var);

template <typename VarType>
framework::proto::VarType::Type GetDataType(std::shared_ptr<VarType> var);

template <typename VarType>
const std::shared_ptr<VariableWrapper>& GetVariableWrapper(
    const std::shared_ptr<VarType>& var);
}  // namespace imperative
}  // namespace paddle
