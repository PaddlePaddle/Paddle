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

// This file contains model format related operations, such as load a model,
// parse an operator definitions and so on.

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/core/tensor.h"
#include "paddle/fluid/lite/core/variable.h"

namespace paddle {
namespace lite {

// Read a __model__ file.
std::unique_ptr<framework::proto::ProgramDesc> LoadProgram(
    const std::string& path);

// Read a single file containing all the parameters.
void LoadParams(const std::string& path);

// Load a single parameter to an output tensor.
void LoadParam(const std::string& path, Variable* out);

// Read a model and files of parameters.
void LoadModel(const std::string& model_dir, Scope* scope,
               framework::proto::ProgramDesc* prog);

}  // namespace lite
}  // namespace paddle
