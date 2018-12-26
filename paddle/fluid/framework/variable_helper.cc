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

#include "paddle/fluid/framework/variable_helper.h"

#include <vector>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
void InitializeVariable(Variable* var, proto::VarType::Type var_type) {
  if (var_type == proto::VarType::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == proto::VarType::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == proto::VarType::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::STEP_SCOPES) {
    var->GetMutable<std::vector<framework::Scope*>>();
  } else if (var_type == proto::VarType::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == proto::VarType::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else if (var_type == proto::VarType::PLACE_LIST) {
    var->GetMutable<platform::PlaceList>();
  } else if (var_type == proto::VarType::READER) {
    var->GetMutable<ReaderHolder>();
  } else if (var_type == proto::VarType::RAW) {
    // GetMutable will be called in operator
  } else {
    PADDLE_THROW(
        "Variable type %d is not in "
        "[LOD_TENSOR, SELECTED_ROWS, FEED_MINIBATCH, FETCH_LIST, "
        "LOD_RANK_TABLE, PLACE_LIST, READER, RAW]",
        var_type);
  }
}

void CopyVariable(const std::string& var_name,
                  const framework::Scope& src_scope,
                  framework::Scope* dst_scope) {
  auto* src_var = src_scope.FindVar(var_name);
  PADDLE_ENFORCE(src_var != nullptr, "");
  platform::CPUPlace cpu;
  auto* dst_var = dst_scope->Var(var_name);
  if (src_var->IsType<framework::LoDTensor>()) {
    auto& src_tensor = src_var->Get<framework::LoDTensor>();
    auto* dst_tensor = dst_var->GetMutable<framework::LoDTensor>();

    dst_tensor->set_lod(src_tensor.lod());
    paddle::framework::TensorCopy(src_tensor, cpu, dst_tensor);
  } else if (src_var->IsType<framework::SelectedRows>()) {
    auto& src_slr = src_var->Get<framework::SelectedRows>();
    auto* dst_slr = dst_var->GetMutable<framework::SelectedRows>();
    dst_slr->set_rows(src_slr.rows());
    dst_slr->set_height(src_slr.height());
    paddle::framework::TensorCopy(src_slr.value(), cpu,
                                  dst_slr->mutable_value());
  } else {
    PADDLE_THROW("Serialize does not support type: %s",
                 typeid(src_var->Type()).name());
  }
}

}  // namespace framework
}  // namespace paddle
