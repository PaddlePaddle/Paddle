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

#include "paddle/fluid/distributed/collective/custom_ccl_tools.h"

namespace paddle {
namespace distributed {

phi::ccl::CCLReduceOp ToXCCLRedType(ReduceOp reduction) {
  static const std::map<ReduceOp, phi::ccl::CCLReduceOp> red_type = {
      {ReduceOp::MIN, phi::ccl::CCLReduceOp::MIN},
      {ReduceOp::MAX, phi::ccl::CCLReduceOp::MAX},
      {ReduceOp::SUM, phi::ccl::CCLReduceOp::SUM},
      {ReduceOp::PRODUCT, phi::ccl::CCLReduceOp::PRODUCT},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(
      it != red_type.end(),
      true,
      common::errors::InvalidArgument("Invalid CustomCCL reduction. "
                                      "Must be Min | Max | Prod | Sum"));
  return it->second;
}

}  //  namespace distributed
}  //  namespace paddle
