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

#include "paddle/fluid/distributed/collective/nccl_tools.h"

#include <unordered_map>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {

ncclRedOp_t ToNCCLRedType(ReduceOp reduction) {
  static const std::unordered_map<ReduceOp, ncclRedOp_t> red_type = {
      {ReduceOp::MIN, ncclMin},
      {ReduceOp::MAX, ncclMax},
      {ReduceOp::SUM, ncclSum},
      {ReduceOp::PRODUCT, ncclProd},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(),
                    true,
                    phi::errors::InvalidArgument(
                        "Invalid nccl reduction. Must be ncclMin | ncclMax | "
                        "ncclProd | ncclSum"));
  return it->second;
}

std::string SerializeNCCLUniqueId(const ncclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (auto i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string GetCommDebugString(ncclComm_t comm) {
  int dev_id;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclCommCuDevice(comm, &dev_id));
  int rank;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclCommUserRank(comm, &rank));
  int nranks;
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclCommCount(comm, &nranks));
  std::stringstream ss;
  ss << std::hex << comm;
  return "[" + ss.str() + " dev_id=" + std::to_string(dev_id) +
         " rank=" + std::to_string(rank) + " nranks=" + std::to_string(nranks) +
         "]";
}

std::string NCCLDTypeToString(ncclDataType_t dtype) {
#define PD_NCCL_DTYPE_TO_STR(__nccl_dtype, __str_dtype) \
  if (dtype == __nccl_dtype) return __str_dtype;

  PD_NCCL_DTYPE_TO_STR(ncclFloat, "float32");
  PD_NCCL_DTYPE_TO_STR(ncclFloat32, "float32");
  PD_NCCL_DTYPE_TO_STR(ncclHalf, "float16");
  PD_NCCL_DTYPE_TO_STR(ncclFloat16, "float16");
#if NCCL_VERSION_CODE >= 21000
  PD_NCCL_DTYPE_TO_STR(ncclBfloat16, "bfloat16");
#endif
  PD_NCCL_DTYPE_TO_STR(ncclDouble, "float64");
  PD_NCCL_DTYPE_TO_STR(ncclFloat64, "float64");

  PD_NCCL_DTYPE_TO_STR(ncclInt8, "int8");
  PD_NCCL_DTYPE_TO_STR(ncclChar, "int8");
  PD_NCCL_DTYPE_TO_STR(ncclUint8, "uint8");
  PD_NCCL_DTYPE_TO_STR(ncclInt32, "int32");
  PD_NCCL_DTYPE_TO_STR(ncclInt, "int32");
  PD_NCCL_DTYPE_TO_STR(ncclUint32, "uint32");
  PD_NCCL_DTYPE_TO_STR(ncclInt64, "int64");
  PD_NCCL_DTYPE_TO_STR(ncclUint64, "uint64");

#undef PD_NCCL_DTYPE_TO_STR
  PADDLE_THROW(phi::errors::InvalidArgument(
      "This datatype %d in nccl is not supported.", static_cast<int>(dtype)));
}

std::string NCCLRedTypeToString(ncclRedOp_t op) {
  if (op == ncclSum) return "SUM";
  if (op == ncclProd) return "PROD";
  if (op == ncclMin) return "MIN";
  if (op == ncclMax) return "MAX";
#if NCCL_VERSION_CODE >= 21000
  if (op == ncclAvg) return "AVG";
#endif

  return "UDF_" + std::to_string(op);
}

}  //  namespace distributed
}  //  namespace paddle
