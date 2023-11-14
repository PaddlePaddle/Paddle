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

#include "glog/logging.h"

PHI_DECLARE_bool(enable_pir_in_executor);
PHI_DECLARE_bool(enable_pir_api);

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {

class Variable;

void SetVariable(Scope* scope,
                 const phi::DenseTensor& input,
                 const std::string& var_name) {
  bool is_new_var = (scope->FindVar(var_name) == nullptr);
  // For any new var that is waiting to be created, we are sure it is a
  // DenseTensor as indicated by parameter list ("const phi::DenseTensor&
  // input"), but if a var already exists, we need to check if it is a
  // DenseTensor, if not, throw an error.
  auto target_var = scope->Var(var_name);
  if (is_new_var) {
    auto val = target_var->GetMutable<phi::DenseTensor>();
    val->ShareDataWith(input);
    val->set_lod(input.lod());
  } else {
    if (!target_var->IsType<phi::DenseTensor>()) {
      int dense_tensor_type_id = VarTypeTrait<phi::DenseTensor>::kId;
      int target_var_type_id = target_var->Type();
      std::string err_msg =
          std::string("Variable \"") + var_name + std::string("\" (typeid=") +
          std::to_string(target_var_type_id) + std::string(") ") +
          std::string("is not a phi::DenseTensor (typeid=") +
          std::to_string(dense_tensor_type_id) + std::string(") ") +
          std::string("but here you tried to convert its type. Aborted.");
      PADDLE_THROW(phi::errors::InvalidArgument(err_msg));
    } else {
      // it is a DenseTensor, everything is fine, just need to update its value.
      target_var->Clear();
      auto val = target_var->GetMutable<phi::DenseTensor>();
      val->ShareDataWith(input);
      val->set_lod(input.lod());
    }
  }
}

void SetFeedVariable(Scope* scope,
                     const phi::DenseTensor& input,
                     const std::string& var_name,
                     size_t index) {
  // If var_name Variable is not found in GlobalScope, a new variable will
  // be created.
  VLOG(3) << "SetFeedVariable name=" << var_name << " index=" << index;
  if (FLAGS_enable_pir_in_executor) {
    // shared data with input tensor
    auto feed_ele = scope->Var(var_name);
    if (!feed_ele->IsType<phi::DenseTensor>()) {
      VLOG(3) << "Reset " << var_name << " to phi::DenseTensor";
      feed_ele->Clear();
    }
    auto val = feed_ele->GetMutable<phi::DenseTensor>();
    val->ShareDataWith(input);
    // set lod
    val->set_lod(input.lod());
  } else {
    Variable* g_feed_value = scope->Var(var_name);
    auto& feed_inputs = *(g_feed_value->GetMutable<FeedList>());
    if (index >= feed_inputs.size()) {
      feed_inputs.resize(index + 1);
    }
    // shared data with input tensor
    auto& val = PADDLE_GET(phi::DenseTensor, feed_inputs[index]);
    val.ShareDataWith(input);
    // set lod
    val.set_lod(input.lod());
  }
}

void SetFeedVariable(Scope* scope,
                     const std::vector<std::string>& input,
                     const std::string& var_name,
                     size_t index) {
  // If var_name Variable is not found in GlobalScope, a new variable will
  // be created.
  VLOG(3) << "SetFeedStringVariable name=" << var_name << " index=" << index;
  Variable* g_feed_value = scope->Var(var_name);
  auto& feed_inputs = *(g_feed_value->GetMutable<FeedList>());
  if (index >= feed_inputs.size()) {
    feed_inputs.resize(index + 1);
  }
  // shared data with input tensor
  feed_inputs[index] = Strings(input);
}

FetchType& GetFetchVariable(const Scope& scope,
                            const std::string& var_name,
                            size_t index) {
  // Since we want to fetch FetchType from a variable, the variable must
  // be created alreadly.
  Variable* g_fetch_value = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(g_fetch_value,
                          platform::errors::NotFound(
                              "Variable %s is not found in scope.", var_name));
  PADDLE_ENFORCE_EQ(g_fetch_value->IsType<FetchList>(),
                    true,
                    platform::errors::InvalidArgument(
                        "Only %s can be invoked by GetFetchVariable",
                        typeid(FetchList).name()));
  auto& fetch_outputs = *g_fetch_value->GetMutable<FetchList>();
  auto& tensor = fetch_outputs[index];
  VLOG(3) << "Fetch " << var_name << " with index " << index;
  PADDLE_ENFORCE_LT(index,
                    fetch_outputs.size(),
                    platform::errors::InvalidArgument(
                        "index must less than fetch_outputs size."));
  return tensor;
}

phi::DenseTensor& GetVariableTensor(const Scope& scope,
                                    const std::string& var_name) {
  Variable* var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(var,
                          platform::errors::NotFound(
                              "Variable %s is not found in scope.", var_name));
  PADDLE_ENFORCE_EQ(var->IsType<phi::DenseTensor>(),
                    true,
                    platform::errors::InvalidArgument(
                        "Only support DenseTensor in GetVariableTensor now."));
  return *var->GetMutable<phi::DenseTensor>();
}

}  // namespace framework
}  // namespace paddle
