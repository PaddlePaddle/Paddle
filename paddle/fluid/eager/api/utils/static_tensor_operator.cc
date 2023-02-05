//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/utils/static_tensor_operator.h"

#include "glog/logging.h"
#include "paddle/fluid/prim/api/generated/prim_api/prim_generated_api.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

namespace paddle {

namespace experimental {
using DescTensor = paddle::prim::DescTensor;

StaticTensorOperator& StaticTensorOperator::Instance() {
  static StaticTensorOperator g_static_op;
  return g_static_op;
}

Tensor StaticTensorOperator::multiply(const Tensor& x, const Tensor& y) {
  VLOG(1) << "DEBUG dispatched in static mode";
  return paddle::prim::multiply<DescTensor>(x, y);
}

}  // namespace experimental
}  // namespace paddle
