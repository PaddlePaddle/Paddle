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

#include "paddle/phi/core/distributed/nccl_tools.h"

#include <unordered_map>

#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"

// #if NCCL_VERSION_CODE >= 21300
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
// #endif

namespace phi {
namespace distributed {

mcclRedOp_t ToNCCLRedType(ReduceOp reduction) {
  static const std::unordered_map<ReduceOp, mcclRedOp_t> red_type = {
      {ReduceOp::MIN, mcclMin},
      {ReduceOp::MAX, mcclMax},
      {ReduceOp::SUM, mcclSum},
      {ReduceOp::PRODUCT, mcclProd},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(),
                    true,
                    phi::errors::InvalidArgument(
                        "Invalid nccl reduction. Must be mcclMin | mcclMax | "
                        "mcclProd | mcclSum"));
  return it->second;
}

std::string SerializeNCCLUniqueId(const mcclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (auto i = 0; i < MCCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string NCCLDTypeToString(mcclDataType_t dtype) {
#define PD_NCCL_DTYPE_TO_STR(__nccl_dtype, __str_dtype) \
  if (dtype == __nccl_dtype) return __str_dtype;
  PD_NCCL_DTYPE_TO_STR(mcclFloat, "float32");
  PD_NCCL_DTYPE_TO_STR(mcclFloat32, "float32");
  PD_NCCL_DTYPE_TO_STR(mcclHalf, "float16");
  PD_NCCL_DTYPE_TO_STR(mcclFloat16, "float16");
// // #if NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000
//   PD_NCCL_DTYPE_TO_STR(mcclBfloat16, "bfloat16");
// // #endif
  PD_NCCL_DTYPE_TO_STR(mcclDouble, "float64");
  PD_NCCL_DTYPE_TO_STR(mcclFloat64, "float64");

  PD_NCCL_DTYPE_TO_STR(mcclInt8, "int8");
  PD_NCCL_DTYPE_TO_STR(mcclChar, "int8");
  PD_NCCL_DTYPE_TO_STR(mcclUint8, "uint8");
  PD_NCCL_DTYPE_TO_STR(mcclInt32, "int32");
  PD_NCCL_DTYPE_TO_STR(mcclInt, "int32");
  PD_NCCL_DTYPE_TO_STR(mcclUint32, "uint32");
  PD_NCCL_DTYPE_TO_STR(mcclInt64, "int64");
  PD_NCCL_DTYPE_TO_STR(mcclUint64, "uint64");

#undef PD_NCCL_DTYPE_TO_STR
  PADDLE_THROW(phi::errors::InvalidArgument(
      "This datatype %d in nccl is not supported.", static_cast<int>(dtype)));
}

std::string NCCLRedTypeToString(mcclRedOp_t op) {
  if (op == mcclSum) return "SUM";
  if (op == mcclProd) return "PROD";
  if (op == mcclMin) return "MIN";
  if (op == mcclMax) return "MAX";
// #if NCCL_VERSION_CODE >= 21000
  if (op == mcclAvg) return "AVG";
// #endif
  return "UDF_" + std::to_string(op);
}

}  //  namespace distributed
}  // namespace phi
