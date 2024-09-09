// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/bkcl_tools.h"

namespace paddle {
namespace distributed {

BKCLOp ToBKCLRedType(ReduceOp reduction) {
  static const std::map<ReduceOp, BKCLOp> red_type = {
      {ReduceOp::MIN, BKCL_MIN},
      {ReduceOp::MAX, BKCL_MAX},
      {ReduceOp::SUM, BKCL_ADD},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(),
                    true,
                    common::errors::InvalidArgument(
                        "Invalid bkcl reduction. Must be BKCL_MIN | BKCL_MAX | "
                        "BKCL_ADD"));
  return it->second;
}

std::string SerializeBKCLUniqueId(const BKCLUniqueId& bkclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&bkclID);
  std::ostringstream oss;
  for (auto i = 0; i < BKCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string BKCLDTypeToString(BKCLDataType dtype) {
#define PD_BKCL_DTYPE_TO_STR(__bkcl_dtype, __str_dtype) \
  if (dtype == __bkcl_dtype) return __str_dtype;
  PD_BKCL_DTYPE_TO_STR(BKCL_FLOAT, "float32");
  PD_BKCL_DTYPE_TO_STR(BKCL_FLOAT16, "float16");
  PD_BKCL_DTYPE_TO_STR(BKCL_BFLOAT16, "bfloat16");
  PD_BKCL_DTYPE_TO_STR(BKCL_FLOAT64, "float64");
  PD_BKCL_DTYPE_TO_STR(BKCL_UINT8, "uint8");
  PD_BKCL_DTYPE_TO_STR(BKCL_INT32, "int32");
  PD_BKCL_DTYPE_TO_STR(BKCL_INT64, "int64");

#undef PD_BKCL_DTYPE_TO_STR
  PADDLE_THROW(common::errors::InvalidArgument(
      "This datatype %d in bkcl is not supported.", static_cast<int>(dtype)));
}

std::string BKCLRedTypeToString(BKCLOp op) {
  if (op == BKCL_ADD) return "SUM";
  if (op == BKCL_PRODUCT) return "PROD";
  if (op == BKCL_MIN) return "MIN";
  if (op == BKCL_MAX) return "MAX";
  return "UDF_" + std::to_string(op);
}

}  //  namespace distributed
}  //  namespace paddle
