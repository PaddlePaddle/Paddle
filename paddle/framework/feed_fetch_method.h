/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/scope.h"
#include "paddle/framework/variable.h"

namespace paddle {
namespace framework {

template <typename T>
void SetFeedVariable(const LoDTensor& input, const std::string& var_name,
                     size_t index) {
  // If var_name Variable is not found in GlobalScope, a new variable will
  // be created.
  Variable* g_feed_value = GetGlobalScope().Var(var_name);
  auto& feed_inputs =
      *(g_feed_value->GetMutable<std::vector<paddle::framework::LoDTensor>>());
  if (index >= feed_inputs.size()) {
    feed_inputs.resize(index + 1);
  }
  // shared data with input tensor
  feed_inputs[index].ShareDataWith<T>(input);
  // set lod
  feed_inputs[index].set_lod(input.lod());
}

LoDTensor& GetFetchVariable(const std::string& var_name, size_t index) {
  // Since we want to fetch LodTensor from a variable, the variable must
  // be created alreadly.
  Variable* g_fetch_value = GetGlobalScope().FindVar(var_name);
  auto& fetch_outputs =
      *(g_fetch_value->GetMutable<std::vector<paddle::framework::LoDTensor>>());
  PADDLE_ENFORCE_LT(index, fetch_outputs.size());
  return fetch_outputs[index];
}

}  // namespace framework
}  // namespace paddle
