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

#include "paddle/phi/api/include/phi_tensor_operator.h"
#include "glog/logging.h"
#include "paddle/phi/api/include/api.h"

namespace paddle {

namespace experimental {

PhiTensorOperator& PhiTensorOperator::Instance() {
  static PhiTensorOperator g_phi_op;
  return g_phi_op;
}

Tensor PhiTensorOperator::multiply(const Tensor& x, const Tensor& y) {
  VLOG(1) << "DEBUG dispatched in phi mode";
  return paddle::experimental::multiply(x, y);
}

}  // namespace experimental
}  // namespace paddle
