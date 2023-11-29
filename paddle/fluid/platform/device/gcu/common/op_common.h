/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace phi {
class DenseTensor;
class Place;
}  // namespace phi

namespace paddle {

namespace framework {
class Scope;
class BlockDesc;
class ExecutionContext;
class VarDesc;

namespace ir {
class Graph;
}  // namespace ir
}  // namespace framework

namespace platform {
using Place = phi::Place;
}

using framework::BlockDesc;
using framework::ExecutionContext;
using framework::Scope;
using framework::Variable;
using framework::ir::Graph;
using LoDTensor = phi::DenseTensor;
using VarNameValuePair = std::pair<std::string, Variable *>;

namespace operators {
namespace gcu {

void ShareVarsIntoScope(const std::vector<VarNameValuePair> &vars,
                        Scope *scope);
void ShareVarsFromScope(const std::vector<VarNameValuePair> &vars,
                        const std::vector<std::string> var_names,
                        const BlockDesc &global_block,
                        Scope *scope);
void GetTensorsByNameFromScope(
    Scope &scope,                            // NOLINT
    std::vector<LoDTensor *> &tensors,       // NOLINT
    std::vector<std::string> &tensor_names,  // NOLINT
    const std::vector<VarNameValuePair> &vars,
    const std::map<std::string, framework::VarDesc *> graph_var_nodes,
    const bool skip_zero_memory = true);

void CompileAndRunGraph(const platform::Place ctx_place,
                        const std::string &program_key,
                        std::vector<std::string> &input_names,   // NOLINT
                        std::vector<std::string> &output_names,  // NOLINT
                        const std::vector<LoDTensor *> &inputs,
                        std::vector<LoDTensor *> &outputs,  // NOLINT
                        Scope &scope,                       // NOLINT
                        const std::shared_ptr<Graph> &graph,
                        const int train_flag = 1);
}  // namespace gcu

}  // namespace operators
}  // namespace paddle
