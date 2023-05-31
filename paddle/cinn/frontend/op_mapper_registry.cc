// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/op_mapper_registry.h"

#include "paddle/cinn/frontend/paddle/cpp/var_desc.h"

namespace cinn {
namespace frontend {

void OpMapperContext::AddVar(const std::string& origin_name,
                             const Variable& var,
                             bool can_inplace) const {
  CHECK(can_inplace || !var_map_->count(origin_name))
      << "Duplicate variable \"" << origin_name << "\" found, whose id is "
      << var_map_->at(origin_name)->id;
  if (var_map_->count(origin_name)) {
    VLOG(1) << "The Paddle inplace output var \""
            << origin_name + paddle::InplaceOutSuffix
            << "\" is mapped to CINN var \"" << var->id << "\" with shape=["
            << cinn::utils::Join(var->shape, ", ") << "], dtype=" << var->type
            << ". The input var \"" << origin_name << "\" still mapped to \""
            << var_map_->at(origin_name)->id << "\"";
  } else {
    VLOG(1) << "The Paddle var \"" << origin_name
            << "\" is mapped to CINN var \"" << var->id << "\" with shape=["
            << cinn::utils::Join(var->shape, ", ") << "], dtype=" << var->type;
  }
  (*var_map_)[origin_name] = var;
}

void OpMapperContext::AddVarModelToProgram(const std::string& name,
                                           const std::string& id,
                                           bool can_inplace) const {
  CHECK(!id.empty()) << "Paddle name [" << name
                     << "]'s program id is empty ! Please check.";
  if (!var_model_to_program_map_->count(name)) {
    (*var_model_to_program_map_)[name] = id;
    VLOG(4) << "Paddle name [" << name << "] map to program id " << id;
  } else {
    CHECK(can_inplace) << "Duplicate variable [" << name
                       << "] found, whose id is "
                       << var_model_to_program_map_->at(name);

    const auto& inplace_out_name = name + paddle::InplaceOutSuffix;
    (*var_model_to_program_map_)[inplace_out_name] = id;

    VLOG(4) << "Paddle name [" << name << "] 's trick output ["
            << inplace_out_name << "] map to program id [" << id << "]";
  }
}

void OpMapperContext::AddFetchVarName(const std::string& name) const {
  fetch_var_names_->insert(name);
}

Variable OpMapperContext::GetVar(const std::string& origin_name) const {
  auto it = var_map_->find(origin_name);
  if (it != var_map_->end()) return it->second;

  const auto& name = cinn::utils::TransValidVarName(origin_name);
  CheckVarNameValid(name);

  auto* var = scope_.FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    Variable local_var;
    local_var.set_id(name);
    local_var->shape = tensor->shape().data();
    local_var->type = tensor->type();
    AddVar(origin_name, local_var);
    return local_var;
  }

  LOG(FATAL) << "No var called [" << origin_name << "] exists";
  return Variable();
}

void OpMapperContext::AddFeedInfo(const std::string& name,
                                  const FeedInfo& info) {
  CHECK(!feed_info_map_.count(name))
      << "Duplicate variable info [" << name << "] found";
  feed_info_map_[name] = info;
}

const OpMapperContext::FeedInfo& OpMapperContext::GetFeedInfo(
    const std::string& name) const {
  CHECK(feed_info_map_.count(name))
      << "No variable info called [" << name << "] exists";
  return feed_info_map_.at(name);
}

}  // namespace frontend
}  // namespace cinn
