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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/frontend/paddle/cpp/program_desc.h"
#include "paddle/cinn/frontend/paddle/framework.pb.h"
#include "paddle/cinn/frontend/paddle/pb/block_desc.h"
#include "paddle/cinn/frontend/paddle/pb/op_desc.h"
#include "paddle/cinn/frontend/paddle/pb/program_desc.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/hlir/framework/tensor.h"

namespace cinn::frontend::paddle {
namespace framework_proto = ::cinn::frontend::paddle::proto;

// Read a model and files of parameters in pb format.
void LoadModelPb(const std::string& model_dir,
                 const std::string& model_file,
                 const std::string& param_file,
                 hlir::framework::Scope* scope,
                 cpp::ProgramDesc* cpp_prog,
                 bool combined = true,
                 bool model_from_memory = false,
                 const common::Target& target = common::DefaultHostTarget());

// Read a __model__ file.
std::unique_ptr<framework_proto::ProgramDesc> LoadProgram(
    const std::string& path, bool program_from_memory = false);

void LoadLoDTensor(std::istream& is,
                   hlir::framework::Variable* var,
                   const common::Target& target);

// Read a single file containing all the parameters.
void LoadParams(const std::string& path);

// Load a single parameter to an output tensor.
void LoadParam(const std::string& path,
               hlir::framework::Variable* out,
               const common::Target& target);

void LoadCombinedParamsPb(
    const std::string& path,
    hlir::framework::Scope* scope,
    const pb::ProgramDesc& prog,
    bool params_from_memory = false,
    const common::Target& target = common::DefaultHostTarget());

// LoDTensor to ostream
void TensorToStream(std::ostream& os, const hlir::framework::_Tensor_& tensor);
void TensorFromStream(
    std::istream& is,
    hlir::framework::_Tensor_* tensor,
    const common::Target& target = common::DefaultHostTarget());
void ReadBinaryFile(const std::string& filename, std::string* contents);

}  // namespace cinn::frontend::paddle
