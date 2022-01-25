// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/legacy/tensor_helper.h"

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/place.h"

namespace egr {
namespace legacy {

void InitializeVariable(paddle::framework::Variable *var,
                        paddle::framework::proto::VarType::Type var_type) {
  if (var_type == paddle::framework::proto::VarType::LOD_TENSOR) {
    var->GetMutable<paddle::framework::LoDTensor>();
  } else if (var_type == paddle::framework::proto::VarType::SELECTED_ROWS) {
    var->GetMutable<paddle::framework::SelectedRows>();
  } else if (var_type == paddle::framework::proto::VarType::FEED_MINIBATCH) {
    var->GetMutable<paddle::framework::FeedList>();
  } else if (var_type == paddle::framework::proto::VarType::FETCH_LIST) {
    var->GetMutable<paddle::framework::FetchList>();
  } else if (var_type == paddle::framework::proto::VarType::STEP_SCOPES) {
    var->GetMutable<std::vector<paddle::framework::Scope *>>();
  } else if (var_type == paddle::framework::proto::VarType::LOD_RANK_TABLE) {
    var->GetMutable<paddle::framework::LoDRankTable>();
  } else if (var_type == paddle::framework::proto::VarType::LOD_TENSOR_ARRAY) {
    var->GetMutable<paddle::framework::LoDTensorArray>();
  } else if (var_type == paddle::framework::proto::VarType::STRINGS) {
    var->GetMutable<paddle::framework::Strings>();
  } else if (var_type == paddle::framework::proto::VarType::VOCAB) {
    var->GetMutable<paddle::framework::Vocab>();
  } else if (var_type == paddle::framework::proto::VarType::PLACE_LIST) {
    var->GetMutable<paddle::platform::PlaceList>();
  } else if (var_type == paddle::framework::proto::VarType::READER) {
    var->GetMutable<paddle::framework::ReaderHolder>();
  } else if (var_type == paddle::framework::proto::VarType::RAW) {
    // GetMutable will be called in operator
  } else {
    PADDLE_THROW(paddle::platform::errors::Unavailable(
        "paddle::framework::Variable type %d is not in "
        "[LOD_TENSOR, SELECTED_ROWS, FEED_MINIBATCH, FETCH_LIST, "
        "LOD_RANK_TABLE, PLACE_LIST, READER, RAW].",
        var_type));
  }
}

void CopyVariable(const paddle::framework::Variable &src_var,
                  paddle::framework::Variable *dst_var) {
  // only support cpu now
  auto cpu_place = paddle::platform::CPUPlace();

  if (src_var.IsType<paddle::framework::LoDTensor>()) {
    auto *tmp_grad_tensor = dst_var->GetMutable<paddle::framework::LoDTensor>();
    auto &src_tensor = src_var.Get<paddle::framework::LoDTensor>();
    tmp_grad_tensor->set_lod(src_tensor.lod());
    paddle::framework::TensorCopy(src_tensor, cpu_place, tmp_grad_tensor);
  } else if (src_var.IsType<paddle::framework::SelectedRows>()) {
    auto &src_slr = src_var.Get<paddle::framework::SelectedRows>();
    auto *tmp_grad_slr = dst_var->GetMutable<paddle::framework::SelectedRows>();
    tmp_grad_slr->set_rows(src_slr.rows());
    tmp_grad_slr->set_height(src_slr.height());
    auto &src_t = src_slr.value();
    auto *dst_t = tmp_grad_slr->mutable_value();
    paddle::framework::TensorCopy(src_t, cpu_place, dst_t);
  } else {
    PADDLE_THROW(paddle::platform::errors::Unavailable(
        "Unknown variable type to copy."));
  }
}
paddle::framework::proto::VarType::Type GetDtypeFromVar(
    const paddle::framework::Variable &var) {
  if (var.IsType<paddle::framework::LoDTensor>()) {
    return var.Get<paddle::framework::LoDTensor>().type();
  } else if (var.IsType<paddle::framework::SelectedRows>()) {
    return var.Get<paddle::framework::SelectedRows>().value().type();
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Variable type is %s, expect LoDTensor or SelectedRows.",
        paddle::framework::ToTypeName(var.Type())));
  }
}
const paddle::platform::Place &GetPlaceFromVar(
    const paddle::framework::Variable &var) {
  if (var.IsType<paddle::framework::LoDTensor>()) {
    return var.Get<paddle::framework::LoDTensor>().place();
  } else if (var.IsType<paddle::framework::SelectedRows>()) {
    return var.Get<paddle::framework::SelectedRows>().place();
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Variable type is %s, expect LoDTensor or SelectedRows.",
        paddle::framework::ToTypeName(var.Type())));
  }
}

}  // namespace legacy
}  // namespace egr
