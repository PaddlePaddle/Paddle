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

#pragma once

#ifndef _WIN32

#include <unistd.h>
#include <cstdint>
#include <set>

namespace paddle {
namespace imperative {

extern void SetLoadProcessPIDs(int64_t key, std::set<pid_t> pids);
extern void EraseLoadProcessPIDs(int64_t key);
extern void SetLoadProcessSignalHandler();
extern void ThrowErrorIfLoadProcessFailed();

}  // namespace imperative
}  // namespace paddle

#endif
