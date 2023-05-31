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

#pragma once

#include "cinn/common/macros.h"
#include "cinn/common/type.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/paddle/cpp/desc_api.h"
#include "cinn/frontend/paddle/cpp/var_desc.h"

namespace cinn {
namespace frontend {
namespace utils {

inline common::Type CppVarType2CommonType(paddle::cpp::VarDescAPI::Type type) {
#define SET_TYPE_CASE_ITEM(v_type, c_type)    \
  case paddle::cpp::VarDescAPI::Type::v_type: \
    return common::c_type();                  \
    break;

  static std::vector<std::string> var_type_names_ = {"BOOL",
                                                     "INT16",
                                                     "INT32",
                                                     "INT64",
                                                     "FP16",
                                                     "FP32",
                                                     "FP64",
                                                     "LOD_TENSOR",
                                                     "SELECTED_ROWS",
                                                     "FEED_MINIBATCH",
                                                     "FETCH_LIST",
                                                     "STEP_SCOPES",
                                                     "LOD_RANK_TABLE",
                                                     "LOD_TENSOR_ARRAY",
                                                     "PLACE_LIST",
                                                     "READER",
                                                     "",
                                                     "RAW",
                                                     "TUPLE",
                                                     "SIZE_T",
                                                     "UINT8",
                                                     "INT8",
                                                     "BF16",
                                                     "COMPLEX64",
                                                     "COMPLEX128",
                                                     "STRING",
                                                     "STRINGS",
                                                     "VOCAB",
                                                     "FEED_LIST",
                                                     "PSTRING",
                                                     "SPARSE_COO",
                                                     "SPARSE_CSR"};
  CHECK_LT(static_cast<int>(type), var_type_names_.size())
      << "Unknown VarDesc type: " << static_cast<int>(type);

  switch (type) {
    SET_TYPE_CASE_ITEM(BOOL, Bool)
    SET_TYPE_CASE_ITEM(INT16, I16)
    SET_TYPE_CASE_ITEM(INT32, I32)
    SET_TYPE_CASE_ITEM(INT64, I64)
    SET_TYPE_CASE_ITEM(BF16, BF16)
    SET_TYPE_CASE_ITEM(FP16, F16)
    SET_TYPE_CASE_ITEM(FP32, F32)
    SET_TYPE_CASE_ITEM(FP64, F64)
    SET_TYPE_CASE_ITEM(SIZE_T, UI64)
    SET_TYPE_CASE_ITEM(UINT8, UI8)
    SET_TYPE_CASE_ITEM(INT8, I8)
    SET_TYPE_CASE_ITEM(STRING, String)
    default:
      LOG(FATAL) << "Unknown VarDesc type: "
                 << var_type_names_[static_cast<int>(type)];
  }
#undef SET_DATA_TYPE_CASE_ITEM
  return common::Void();
}

inline OpMapperContext::FeedInfo GetFeedInfoFromDesc(
    const paddle::cpp::VarDesc& desc) {
  OpMapperContext::FeedInfo info;
  for (auto num : desc.GetShape()) {
    info.shape.emplace_back(static_cast<int>(num));
  }
  info.type = CppVarType2CommonType(desc.GetDataType());
  return info;
}

}  // namespace utils
}  // namespace frontend
}  // namespace cinn
