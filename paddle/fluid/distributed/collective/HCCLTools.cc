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

#include "paddle/fluid/distributed/collective/HCCLTools.h"
#include "paddle/fluid/distributed/collective/Types.h"

namespace paddle {
namespace distributed {

HcclReduceOp ToHCCLRedType(ReduceOp reduction) {
  static const std::map<ReduceOp, HcclReduceOp> red_type = {
      {ReduceOp::MIN, HCCL_REDUCE_MIN},
      {ReduceOp::MAX, HCCL_REDUCE_MAX},
      {ReduceOp::SUM, HCCL_REDUCE_SUM},
      {ReduceOp::PRODUCT, HCCL_REDUCE_PROD},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(
      it != red_type.end(), true,
      platform::errors::InvalidArgument("Invalid hccl reduction. "
                                        "Must be Min | Max | Prod | Sum"));
  return it->second;
}

std::string SerializeHCCLUniqueId(const HcclRootInfo& hcclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&hcclID);
  std::ostringstream oss;
  for (size_t i = 0; i < sizeof(hcclID); ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

}  //  namespace distributed
}  //  namespace paddle
