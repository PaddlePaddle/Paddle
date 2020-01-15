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

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {
namespace distributed {

constexpr int64_t kNoPadding = -1;

void prefetchs(const std::vector<std::string>& id_var_names,
               const std::vector<std::string>& out_var_names,
               const std::string& persistable_var_name, const bool backfill,
               const std::vector<std::string>& table_names,
               const std::vector<std::string>& endpoints,
               const std::vector<int64_t>& height_sections,
               const framework::ExecutionContext& context,
               const framework::Scope& scope);

void prefetch(const std::string& id_name, const std::string& out_name,
              const std::string& persistable_var_name, const bool backfill,
              const std::vector<std::string>& table_names,
              const std::vector<std::string>& endpoints,
              const std::vector<int64_t>& height_sections,
              const framework::ExecutionContext& context,
              const framework::Scope& scope);

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
