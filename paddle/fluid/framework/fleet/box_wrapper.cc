// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include <memory>

namespace paddle {
namespace framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
std::shared_ptr<paddle::boxps::BoxPS> BoxWrapper::boxps_ptr_ = nullptr;

int BoxWrapper::PassBegin(const std::set<uint64_t>& feasgin_to_box) const {
  boxps_ptr_->PassBegin(feasgin_to_box);
  return 0;
}

int BoxWrapper::PassEnd() const {
  boxps_ptr_->PassEnd();
  return 0;
}

int BoxWrapper::PullSparsePara(const Scope& scope,
                               const paddle::platform::Place& place,
                               const std::vector<std::vector<uint64_t>>& keys,
                               std::vector<std::vector<float>>* values) {
  boxps_ptr_->PullSparse(keys, values);
  return 0;
}

int BoxWrapper::PushSparseGrad(
    const Scope& scope, const paddle::platform::Place& place,
    const std::vector<std::vector<uint64_t>>& keys,
    const std::vector<std::vector<float>>& grad_values) {
  boxps_ptr_->PushSparse(keys, grad_values);
  return 0;
}
}  // end namespace framework
}  // end namespace paddle
