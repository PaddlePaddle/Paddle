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

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/variable.h"
namespace paddle {
namespace framework {

void InitializeVariable(Variable* var, proto::VarType::Type var_type);
void CopyVariable(const Variable& src_var, Variable* dst_var);

}  // end namespace framework
}  // end namespace paddle
