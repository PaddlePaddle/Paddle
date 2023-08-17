// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/graph_compiler_util.h"

namespace cinn {
namespace hlir {
namespace framework {

void CompilationContext::ApplyTuningResult(
    const auto_schedule::TuningResult& tuning_result) {
  // assign options with TuningResult directly
  groups.assign(tuning_result.subgraphs.begin(), tuning_result.subgraphs.end());
  lowered_funcs.assign(tuning_result.function_groups.begin(),
                       tuning_result.function_groups.end());
}

void CompilationContext::SetAttachedSourceCode(const std::string& code) {
  attached_source_code = code;
}

void CompilationResult::InsertErrorMsgTo(std::vector<std::string>* arr,
                                         const std::string& msg,
                                         const int times) {
  for (int i = 0; i < times; ++i) {
    arr->emplace_back(msg);
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
