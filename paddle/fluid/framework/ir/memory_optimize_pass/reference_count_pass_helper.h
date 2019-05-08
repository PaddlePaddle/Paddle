// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/garbage_collector.h"

namespace paddle {
namespace framework {

class VarDesc;

namespace ir {

using ReferenceCountMap = std::unordered_map<std::string, size_t>;

using AtomicReferenceCountMap =
    std::unordered_map<std::string, std::atomic<size_t>>;

using GarbageCollectorMap =
    std::map<platform::Place, std::unique_ptr<GarbageCollector>>;

const char kGlobalReferenceCount[] = "global_reference_count";
const char kRuntimeReferenceCount[] = "runtime_reference_count";
const char kGarbageCollector[] = "garbage_collector";
const char kAllPlaces[] = "all_places";

using LastLiveOpsOfVars =
    std::unordered_map<std::string,
                       std::unordered_set<details::ComputationOpHandle *>>;
const char kLastLiveOpsOfVars[] = "last_live_ops_of_var";

VarDesc *TryGetLatestVarDesc(const std::vector<details::VarHandle *> &vars);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
