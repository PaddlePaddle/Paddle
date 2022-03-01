// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/utils.h"

#include "paddle/phi/api/all.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/variable.h"

namespace egr {
namespace egr_utils_api {

bool IsLeafTensor(const paddle::experimental::Tensor& target) {
  std::shared_ptr<GradNodeBase> grad_node = EagerUtils::grad_node(target);
  if (std::dynamic_pointer_cast<GradNodeAccumulation>(grad_node)) {
    return true;
  }

  return false;
}

paddle::experimental::Tensor CreateTensorWithValue(
    const phi::DDim& ddim, const paddle::platform::Place& place,
    const phi::DataType& dtype, const phi::DataLayout& layout, float value,
    bool is_leaf) {
  paddle::experimental::Tensor out = paddle::experimental::full(
      phi::vectorize(ddim), paddle::experimental::Scalar(value), dtype,
      phi::TransToPhiBackend(place));

  auto meta = EagerUtils::autograd_meta(&out);
  if (is_leaf) {
    auto accumulation_node = std::make_shared<GradNodeAccumulation>(meta);
    meta->SetGradNode(accumulation_node);
    meta->SetStopGradient(false);
  }

  return out;
}

}  // namespace egr_utils_api
}  // namespace egr
