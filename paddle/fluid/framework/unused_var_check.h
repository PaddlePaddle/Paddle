/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <glog/logging.h>

#include <string>
#include <unordered_set>

#include "paddle/common/flags.h"

namespace paddle {
namespace framework {

class OperatorBase;
class Scope;

std::unordered_set<std::string>* GetThreadLocalUsedVarNameSet();

void LogVarUsageIfUnusedVarCheckEnabled(const std::string& name);
void CheckUnusedVar(const OperatorBase& op, const Scope& scope);

}  // namespace framework
}  // namespace paddle
