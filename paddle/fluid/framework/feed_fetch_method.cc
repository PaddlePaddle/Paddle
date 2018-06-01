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

#include "paddle/fluid/framework/feed_fetch_method.h"
#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {

void SetFeedVariable(Scope* scope, const LoDTensor& input,
                     const std::string& var_name, size_t index) {
  // If var_name Variable is not found in GlobalScope, a new variable will
  // be created.
  VLOG(3) << "SetFeedVariable name=" << var_name << " index=" << index;
  Variable* g_feed_value = scope->Var(var_name);
  auto& feed_inputs =
      *(g_feed_value->GetMutable<std::vector<paddle::framework::LoDTensor>>());
  if (index >= feed_inputs.size()) {
    feed_inputs.resize(index + 1);
  }
  // shared data with input tensor
  feed_inputs[index].ShareDataWith(input);
  // set lod
  feed_inputs[index].set_lod(input.lod());
}

LoDTensor& GetFetchVariable(const Scope& scope, const std::string& var_name,
                            size_t index) {
  // Since we want to fetch LodTensor from a variable, the variable must
  // be created alreadly.
  Variable* g_fetch_value = scope.FindVar(var_name);
  PADDLE_ENFORCE(g_fetch_value->IsType<FeedFetchList>(),
                 "Only %s can be invoked by GetFetchVariable",
                 typeid(FeedFetchList).name());
  auto& fetch_outputs = *g_fetch_value->GetMutable<FeedFetchList>();
  auto& tensor = fetch_outputs[index];
  VLOG(3) << "Fetch " << var_name << " with index " << index
          << " shape= " << tensor.dims();
  PADDLE_ENFORCE_LT(index, fetch_outputs.size());
  return tensor;
}

}  // namespace framework
}  // namespace paddle
