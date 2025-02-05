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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/framework/framework.pb.h"

namespace paddle {
namespace framework {

std::map<int, int> Prune(const proto::ProgramDesc& input,
                         const std::set<std::string>& feed_var_names,
                         proto::ProgramDesc* output);

std::tuple<framework::ProgramDesc, std::map<int, int>> PruneBackward(
    const framework::ProgramDesc& origin);

}  // namespace framework
}  // namespace paddle
