//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/var_handle.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

// all variable in each devices.
// The outside vector is the device vector. Each element of this vector is a
// map from variable name to variables. The variables, who have the same name,
// will have a differsent version. The offset in the
// `std::vector<VarHandle*>` is the version of varaibles.
typedef std::vector<std::unordered_map<std::string, std::vector<VarHandle *>>>
    GraphVars;
constexpr char kGraphVars[] = "vars";

constexpr char kPlaces[] = "places";
constexpr char kLocalScopes[] = "local_scopes";
constexpr char kNCCLCtxs[] = "nccl_ctxs";

// aux variables to represent dependency. Useful to resolve data hazard.
typedef std::unordered_set<VarHandleBase *> GraphDepVars;
constexpr char kGraphDepVars[] = "dep_vars";

// aux variables to represent dependency. Useful to resolve data hazard.
typedef std::unordered_set<std::string> SkipVarSet;
constexpr char kSkipVarSet[] = "skip_var_set";

typedef std::unordered_set<std::string> FusedVars;
constexpr char kFusedVars[] = "fused_vars";
constexpr char kFusedVarNamePrefix[] = "@FUSEDVAR@";

typedef std::string FusedOptType;
constexpr char kFusedOptType[] = "fused_opt_type";

typedef std::string FusedGrads;
constexpr char kFusedGrads[] = "fused_gradients";

typedef std::vector<std::pair<std::string, std::string>> ParamsAndGrads;
constexpr char kParamsAndGrads[] = "params_grads";

typedef std::vector<std::vector<std::pair<std::string, std::string>>>
    GroupGradsAndParams;
constexpr char kGroupGradsAndParams[] = "group_grads_params";

}  // namespace details
}  // namespace framework
}  // namespace paddle
