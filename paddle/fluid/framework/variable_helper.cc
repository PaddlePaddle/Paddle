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

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

void InitializeVariable(Variable *var, proto::VarType::Type var_type) {
  if (var_type == proto::VarType::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == proto::VarType::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == proto::VarType::FEED_MINIBATCH) {
    var->GetMutable<FeedList>();
  } else if (var_type == proto::VarType::FETCH_LIST) {
    var->GetMutable<FetchList>();
  } else if (var_type == proto::VarType::STEP_SCOPES) {
    var->GetMutable<std::vector<framework::Scope *>>();
  } else if (var_type == proto::VarType::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == proto::VarType::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else if (var_type == proto::VarType::STRINGS) {
    var->GetMutable<Strings>();
  } else if (var_type == proto::VarType::VOCAB) {
    var->GetMutable<Vocab>();
  } else if (var_type == proto::VarType::PLACE_LIST) {
    var->GetMutable<platform::PlaceList>();
  } else if (var_type == proto::VarType::READER) {
    var->GetMutable<ReaderHolder>();
  } else if (var_type == proto::VarType::RAW) {
    // GetMutable will be called in operator
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Variable type %d is not in "
        "[LOD_TENSOR, SELECTED_ROWS, FEED_MINIBATCH, FETCH_LIST, "
        "LOD_RANK_TABLE, PLACE_LIST, READER, RAW].",
        var_type));
  }
}

void CopyVariable(const Variable &src_var, Variable *dst_var) {
  // only support cpu now
  auto cpu_place = platform::CPUPlace();

  if (src_var.IsType<framework::LoDTensor>()) {
    auto *tmp_grad_tensor = dst_var->GetMutable<framework::LoDTensor>();
    auto &src_tensor = src_var.Get<framework::LoDTensor>();
    tmp_grad_tensor->set_lod(src_tensor.lod());
    framework::TensorCopy(src_tensor, cpu_place, tmp_grad_tensor);
  } else if (src_var.IsType<framework::SelectedRows>()) {
    auto &src_slr = src_var.Get<framework::SelectedRows>();
    auto *tmp_grad_slr = dst_var->GetMutable<framework::SelectedRows>();
    tmp_grad_slr->set_rows(src_slr.rows());
    tmp_grad_slr->set_height(src_slr.height());
    auto &src_t = src_slr.value();
    auto *dst_t = tmp_grad_slr->mutable_value();
    framework::TensorCopy(src_t, cpu_place, dst_t);
  } else {
    PADDLE_THROW(
        platform::errors::Unavailable("Unknown variable type to copy."));
  }
}

}  // namespace framework
}  // namespace paddle
