// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// This file provides some dependency adding function to handle the implicit
// dependency that cannot be explicitly expresed by a Program. It is a
// compromise of the incomplete expression ability of the Program. Do not add
// too many functions here at will, that will bring great burden to the
// Interpretercore.

// TODO(Ruibiao):
// 1. Move other dependency adding codes from interpretercore_util.cc to
// dependency_utils.cc
// 2. Move other Interpretercore related codes to directory
// new_executor/interpreter
// 3. Try to remove parameter op_happens_before from the dependency adding
// function

#pragma once

#include <map>
#include <vector>

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"

namespace paddle {
namespace framework {
namespace interpreter {

// equivalent to add_reader_dependency_pass
void AddDependencyForReadOp(
    const std::vector<Instruction>& vec_instruction,
    std::map<int, std::list<int>>* downstream_map,
    const std::vector<std::vector<bool>>* op_happens_before = nullptr);

void AddDownstreamOp(
    int prior_op_idx,
    int posterior_op_idx,
    std::map<int, std::list<int>>* op_downstream_map,
    const std::vector<std::vector<bool>>* op_happens_before = nullptr);

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
