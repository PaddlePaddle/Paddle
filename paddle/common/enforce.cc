/* Copyright (c) 2013 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/common/enforce.h"

#include <array>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include "paddle/utils/flags.h"

PD_DECLARE_int32(call_stack_level);
namespace common {
namespace enforce {

int GetCallStackLevel() { return FLAGS_call_stack_level; }
}  // namespace enforce
}  // namespace common
