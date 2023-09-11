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
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"

namespace paddle {
namespace primitive {
template <>
void set_output<LazyTensor>(const paddle::Tensor& x_tmp, paddle::Tensor* x) {
  x->set_impl(x_tmp.impl());
}

std::vector<std::vector<Tensor>> ConstructVjpResultByStopGradients(
    const std::vector<std::vector<Tensor>>& outputs,
    const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<Tensor>> vjp_results(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    vjp_results[i].reserve(outputs[i].size());
    for (size_t j = 0; j < outputs[i].size(); ++j) {
      if (stop_gradients[i][j]) {
        // Use Tensor's impl is nullptr to indicate it has no gradient
        vjp_results[i].emplace_back(Tensor());
      } else {
        vjp_results[i].emplace_back(outputs[i][j]);
      }
    }
  }
  return vjp_results;
}

}  // namespace primitive
}  // namespace paddle
