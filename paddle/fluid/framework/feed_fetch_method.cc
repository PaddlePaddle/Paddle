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
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

void SetFeedVariable(Scope* scope, const LoDTensor& input,
                     const std::string& var_name, size_t index) {
  // If var_name Variable is not found in GlobalScope, a new variable will
  // be created.
  VLOG(3) << "SetFeedVariable name=" << var_name << " index=" << index;
  Variable* g_feed_value = scope->Var(var_name);
  auto& feed_inputs = *(g_feed_value->GetMutable<FeedList>());
  if (index >= feed_inputs.size()) {
    feed_inputs.resize(index + 1);
  }
  // shared data with input tensor
  feed_inputs[index].ShareDataWith(input);
  // set lod
  feed_inputs[index].set_lod(input.lod());
}

FetchType& GetFetchVariable(const Scope& scope, const std::string& var_name,
                            size_t index) {
  // Since we want to fetch FetchType from a variable, the variable must
  // be created alreadly.
  Variable* g_fetch_value = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(g_fetch_value,
                          platform::errors::NotFound(
                              "Variable %s is not found in scope.", var_name));
  PADDLE_ENFORCE_EQ(g_fetch_value->IsType<FetchList>(), true,
                    platform::errors::InvalidArgument(
                        "Only %s can be invoked by GetFetchVariable",
                        typeid(FetchList).name()));
  auto& fetch_outputs = *g_fetch_value->GetMutable<FetchList>();
  auto& tensor = fetch_outputs[index];
  VLOG(3) << "Fetch " << var_name << " with index " << index;
  PADDLE_ENFORCE_LT(index, fetch_outputs.size(),
                    platform::errors::InvalidArgument(
                        "index must less than fetch_outputs size."));
  return tensor;
}

LoDTensor& GetVariableTensor(const Scope& scope, const std::string& var_name) {
  Variable* var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::NotFound("Variable %s is not found in scope.",
                                      var_name));
  PADDLE_ENFORCE_EQ(var->IsType<LoDTensor>(), true,
                    platform::errors::InvalidArgument(
                        "Only support lod tensor in GetVariableTensor now."));
  return *var->GetMutable<LoDTensor>();
}

}  // namespace framework
}  // namespace paddle
