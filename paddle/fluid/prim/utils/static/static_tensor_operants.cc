// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/prim/utils/static/static_tensor_operants.h"

#include "glog/logging.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

namespace paddle {

namespace prim {
using DescTensor = paddle::prim::DescTensor;

Tensor StaticTensorOperants::multiply(const Tensor& x, const Tensor& y) {
  return paddle::prim::multiply<DescTensor>(x, y);
}

}  // namespace prim
}  // namespace paddle
