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

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/paddle/cpp/desc_api.h"
#include "paddle/cinn/frontend/paddle/cpp/var_desc.h"

namespace cinn {
namespace frontend {
namespace utils {

inline common::Type CppVarType2CommonType(paddle::cpp::VarDescAPI::Type type) {
#define SET_TYPE_CASE_ITEM(v_type, c_type)    \
  case paddle::cpp::VarDescAPI::Type::v_type: \
    return common::c_type();                  \
    break;

  static std::vector<std::string> var_type_names_ = {"BOOL",              // 0
                                                     "INT16",             // 1
                                                     "INT32",             // 2
                                                     "INT64",             // 3
                                                     "FP16",              // 4
                                                     "FP32",              // 5
                                                     "FP64",              // 6
                                                     "LOD_TENSOR",        // 7
                                                     "SELECTED_ROWS",     // 8
                                                     "FEED_MINIBATCH",    // 9
                                                     "FETCH_LIST",        // 10
                                                     "STEP_SCOPES",       // 11
                                                     "LOD_RANK_TABLE",    // 12
                                                     "LOD_TENSOR_ARRAY",  // 13
                                                     "PLACE_LIST",        // 14
                                                     "READER",            // 15
                                                     "",
                                                     "RAW",          // 17
                                                     "TUPLE",        // 18
                                                     "SIZE_T",       // 19
                                                     "UINT8",        // 20
                                                     "INT8",         // 21
                                                     "BF16",         // 22
                                                     "COMPLEX64",    // 23
                                                     "COMPLEX128",   // 24
                                                     "STRING",       // 25
                                                     "STRINGS",      // 26
                                                     "VOCAB",        // 27
                                                     "FEED_LIST",    // 28
                                                     "PSTRING",      // 29
                                                     "SPARSE_COO",   // 30
                                                     "SPARSE_CSR"};  // 31
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
    // The paddle's phi::DataType::UNDEFINED is mapped into ProtoDataType::RAW,
    // so here need convert back to unkown type.
    SET_TYPE_CASE_ITEM(RAW, Type)
    default:
      LOG(FATAL) << "Unknown VarDesc type: "
                 << var_type_names_[static_cast<int>(type)] << "("
                 << static_cast<int>(type) << ")";
  }
#undef SET_DATA_TYPE_CASE_ITEM
  return common::Type();
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
