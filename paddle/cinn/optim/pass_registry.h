// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <string>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/lowered_func.h"

namespace cinn {
namespace optim {

// A PassAction encapsulates running a named built-in pass on a LoweredFunc.
// It hides the differences between FuncPass, BlockPass, StmtPass, and
// non-PassManager passes (e.g. Simplify, MapExternCall).
using PassAction = std::function<void(ir::LoweredFunc, const Target&)>;

// Returns the default GPU pass pipeline (ordered list of pass names).
// This matches the current NVGPU behavior in optimize.cc.
std::vector<std::string> GetDefaultGpuPassPipeline();

// Look up a built-in pass action by name.
// Returns nullptr if the name is not a built-in pass (i.e. it is a vendor
// custom pass that should be routed to ApplyCustomPass).
const PassAction* LookupBuiltinPass(const std::string& name);

}  // namespace optim
}  // namespace cinn
