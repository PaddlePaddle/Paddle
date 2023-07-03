// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/frontend/paddle/cpp/block_desc.h"
#include "paddle/cinn/frontend/paddle/cpp/desc_api.h"
#include "paddle/cinn/frontend/paddle/cpp/op_desc.h"
#include "paddle/cinn/frontend/paddle/cpp/program_desc.h"
#include "paddle/cinn/frontend/paddle/cpp/var_desc.h"

namespace cinn::frontend::paddle {

/// Transform an VarDesc from VarDescType to cpp format.
template <typename VarDescType>
void TransformVarDescAnyToCpp(const VarDescType& any_desc,
                              cpp::VarDesc* cpp_desc);

/// Transform an VarDesc from cpp to VarDescType format.
template <typename VarDescType>
void TransformVarDescCppToAny(const cpp::VarDesc& cpp_desc,
                              VarDescType* any_desc);

/// Transform an OpDesc from OpDescType to cpp format.
template <typename OpDescType>
void TransformOpDescAnyToCpp(const OpDescType& any_desc, cpp::OpDesc* cpp_desc);

/// Transform an OpDesc from cpp to OpDescType format.
template <typename OpDescType>
void TransformOpDescCppToAny(const cpp::OpDesc& cpp_desc, OpDescType* any_desc);

/// Transform an BlockDesc from BlockDescType to cpp format.
template <typename BlockDescType>
void TransformBlockDescAnyToCpp(const BlockDescType& any_desc,
                                cpp::BlockDesc* cpp_desc);

/// Transform an BlockDesc from cpp to BlockDescType format.
template <typename BlockDescType>
void TransformBlockDescCppToAny(const cpp::BlockDesc& cpp_desc,
                                BlockDescType* any_desc);

/// Transform an ProgramDesc from ProgramDescType to cpp format.
template <typename ProgramDescType>
void TransformProgramDescAnyToCpp(const ProgramDescType& any_desc,
                                  cpp::ProgramDesc* cpp_desc);

/// Transform an ProgramDesc from cpp to ProgramDescType format.
template <typename ProgramDescType>
void TransformProgramDescCppToAny(const cpp::ProgramDesc& cpp_desc,
                                  ProgramDescType* any_desc);

}  // namespace cinn::frontend::paddle
