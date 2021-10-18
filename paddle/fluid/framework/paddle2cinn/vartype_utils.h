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

#pragma once
#include "glog/logging.h"

#include "cinn/frontend/paddle/cpp/desc_api.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {
namespace utils {

::cinn::frontend::paddle::cpp::VarDescAPI::Type TransformVarTypeToCinn(
    const ::paddle::framework::proto::VarType::Type& type) {
#define SET_TYPE_CASE_ITEM(type__)                                  \
  case ::paddle::framework::proto::VarType::type__:                 \
    return ::cinn::frontend::paddle::cpp::VarDescAPI::Type::type__; \
    break;

  switch (type) {
    SET_TYPE_CASE_ITEM(LOD_TENSOR);
    SET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    SET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    SET_TYPE_CASE_ITEM(SELECTED_ROWS);
    SET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    SET_TYPE_CASE_ITEM(FETCH_LIST);
    SET_TYPE_CASE_ITEM(STEP_SCOPES);
    SET_TYPE_CASE_ITEM(PLACE_LIST);
    SET_TYPE_CASE_ITEM(READER);
    default:
      PADDLE_THROW(platform::errors::NotFound("Cannot found var type"));
  }
#undef SET_TYPE_CASE_ITEM
}

::paddle::framework::proto::VarType::Type TransformVarTypeFromCinn(
    const ::cinn::frontend::paddle::cpp::VarDescAPI::Type& type) {
#define SET_TYPE_CASE_ITEM(type__)                              \
  case ::cinn::frontend::paddle::cpp::VarDescAPI::Type::type__: \
    return ::paddle::framework::proto::VarType::type__;         \
    break;

  switch (type) {
    SET_TYPE_CASE_ITEM(LOD_TENSOR);
    SET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    SET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    SET_TYPE_CASE_ITEM(SELECTED_ROWS);
    SET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    SET_TYPE_CASE_ITEM(FETCH_LIST);
    SET_TYPE_CASE_ITEM(STEP_SCOPES);
    SET_TYPE_CASE_ITEM(PLACE_LIST);
    SET_TYPE_CASE_ITEM(READER);
    default:
      PADDLE_THROW(platform::errors::NotFound("Cannot found var type"));
  }
#undef SET_TYPE_CASE_ITEM
}

::cinn::frontend::paddle::cpp::VarDescAPI::Type TransformVarDataTypeToCinn(
    const ::paddle::framework::proto::VarType::Type& type) {
#define SET_DATA_TYPE_CASE_ITEM(type__)                             \
  case ::paddle::framework::proto::VarType::type__:                 \
    return ::cinn::frontend::paddle::cpp::VarDescAPI::Type::type__; \
    break;

  switch (type) {
    SET_DATA_TYPE_CASE_ITEM(BOOL);
    SET_DATA_TYPE_CASE_ITEM(SIZE_T);
    SET_DATA_TYPE_CASE_ITEM(UINT8);
    SET_DATA_TYPE_CASE_ITEM(INT8);
    SET_DATA_TYPE_CASE_ITEM(INT16);
    SET_DATA_TYPE_CASE_ITEM(INT32);
    SET_DATA_TYPE_CASE_ITEM(INT64);
    SET_DATA_TYPE_CASE_ITEM(FP16);
    SET_DATA_TYPE_CASE_ITEM(FP32);
    SET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      PADDLE_THROW(platform::errors::NotFound("Cannot found var data type"));
  }
#undef SET_DATA_TYPE_CASE_ITEM
}

::paddle::framework::proto::VarType::Type TransformVarDataTypeFromCpp(
    const ::cinn::frontend::paddle::cpp::VarDescAPI::Type& type) {
#define SET_DATA_TYPE_CASE_ITEM(type__)                         \
  case ::cinn::frontend::paddle::cpp::VarDescAPI::Type::type__: \
    return ::paddle::framework::proto::VarType::type__;         \
    break;

  switch (type) {
    SET_DATA_TYPE_CASE_ITEM(BOOL);
    SET_DATA_TYPE_CASE_ITEM(SIZE_T);
    SET_DATA_TYPE_CASE_ITEM(UINT8);
    SET_DATA_TYPE_CASE_ITEM(INT8);
    SET_DATA_TYPE_CASE_ITEM(INT16);
    SET_DATA_TYPE_CASE_ITEM(INT32);
    SET_DATA_TYPE_CASE_ITEM(INT64);
    SET_DATA_TYPE_CASE_ITEM(FP16);
    SET_DATA_TYPE_CASE_ITEM(FP32);
    SET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      PADDLE_THROW(platform::errors::NotFound("Cannot found var data type"));
  }
#undef SET_DATA_TYPE_CASE_ITEM
}

}  // namespace utils
}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
