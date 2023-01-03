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

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

ncclRedOp_t ToNCCLRedType(ReduceOp reduction) {
  static const std::map<ReduceOp, ncclRedOp_t> red_type = {
      {ReduceOp::MIN, ncclMin},
      {ReduceOp::MAX, ncclMax},
      {ReduceOp::SUM, ncclSum},
      {ReduceOp::PRODUCT, ncclProd},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(it != red_type.end(),
                    true,
                    platform::errors::InvalidArgument(
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

}  //  namespace distributed
}  //  namespace paddle
