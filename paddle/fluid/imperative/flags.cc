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

#include "paddle/fluid/imperative/flags.h"
#include "gflags/gflags.h"

DEFINE_uint64(dygraph_debug, 0,
              "Debug level of dygraph. This flag is not "
              "open to users");

/**
 * Example: FLAGS_imperative_check_nan_inf=0, disable imperative NAN/INF check.
 *          FLAGS_imperative_check_nan_inf=1, enable imperative NAN/INF check,
 * raise exception and stop execution when NAN/INF detected at the first time.
 *          FLAGS_imperative_check_nan_inf=2, enable imperative NAN/INF check,
 * do not raise exception but log the message and continue execution.
 */
DEFINE_uint64(
    imperative_check_nan_inf, 0,
    "Checking whether operator produce NAN/INF or not in imperative "
    "mode. It will be extremely slow so please use this flag wisely.");

namespace paddle {
namespace imperative {

bool IsDebugEnabled() { return FLAGS_dygraph_debug != 0; }

uint64_t GetDebugLevel() { return FLAGS_dygraph_debug; }

}  // namespace imperative
}  // namespace paddle
