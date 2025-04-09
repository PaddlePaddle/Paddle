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

#include "paddle/phi/core/distributed/flagcx_tools.h"

#include <unordered_map>

#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {

flagcxRedOp_t ToFlagcxRedType(ReduceOp reduction) {
  static const std::unordered_map<ReduceOp, flagcxRedOp_t> red_type = {
      {ReduceOp::MIN, flagcxMin},
      {ReduceOp::MAX, flagcxMax},
      {ReduceOp::SUM, flagcxSum},
      {ReduceOp::PRODUCT, flagcxProd},
      {ReduceOp::AVG, flagcxAvg},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(
      it != red_type.end(),
      true,
      common::errors::InvalidArgument(
          "Invalid flagcx reduction. Must be flagcxMin | flagcxMax | "
          "flagcxProd | flagcxSum | flagcxAvg."));
  return it->second;
}

std::string SerializeFlagcxUniqueId(const flagcxUniqueId& flagcxID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&flagcxID);
  std::ostringstream oss;
  for (auto i = 0; i < FLAGCX_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string FlagcxDTypeToString(flagcxDataType_t dtype) {
#define PD_FLAGCX_DTYPE_TO_STR(__flagcx_dtype, __str_dtype) \
  if (dtype == __flagcx_dtype) return __str_dtype;
  PD_FLAGCX_DTYPE_TO_STR(flagcxFloat, "float32");
  PD_FLAGCX_DTYPE_TO_STR(flagcxFloat32, "float32");
  PD_FLAGCX_DTYPE_TO_STR(flagcxHalf, "float16");
  PD_FLAGCX_DTYPE_TO_STR(flagcxFloat16, "float16");
  PD_FLAGCX_DTYPE_TO_STR(flagcxBfloat16, "bfloat16");
  PD_FLAGCX_DTYPE_TO_STR(flagcxDouble, "float64");
  PD_FLAGCX_DTYPE_TO_STR(flagcxFloat64, "float64");
  PD_FLAGCX_DTYPE_TO_STR(flagcxInt8, "int8");
  PD_FLAGCX_DTYPE_TO_STR(flagcxChar, "int8");
  PD_FLAGCX_DTYPE_TO_STR(flagcxUint8, "uint8");
  PD_FLAGCX_DTYPE_TO_STR(flagcxInt32, "int32");
  PD_FLAGCX_DTYPE_TO_STR(flagcxInt, "int32");
  PD_FLAGCX_DTYPE_TO_STR(flagcxUint32, "uint32");
  PD_FLAGCX_DTYPE_TO_STR(flagcxInt64, "int64");
  PD_FLAGCX_DTYPE_TO_STR(flagcxUint64, "uint64");

#undef PD_FLAGCX_DTYPE_TO_STR
  PADDLE_THROW(common::errors::InvalidArgument(
      "This datatype %d in flagcx is not supported.", static_cast<int>(dtype)));
}

std::string FlagcxRedTypeToString(flagcxRedOp_t op) {
  if (op == flagcxSum) return "SUM";
  if (op == flagcxProd) return "PROD";
  if (op == flagcxMin) return "MIN";
  if (op == flagcxMax) return "MAX";
  if (op == flagcxAvg) return "AVG";
  return "UDF_" + std::to_string(op);
}

}  //  namespace distributed
}  // namespace phi
