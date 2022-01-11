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

#pragma once

#include "paddle/fluid/eager/grad_node_info.h"

namespace egr {

class GradNodeAccumulation : public GradNodeBase {
 public:
  // Constructor: configure fwd input tensors to grad node
  GradNodeAccumulation() : GradNodeBase(1, 1) { SetDefaultGradInOutMeta(); }

  ~GradNodeAccumulation() override = default;

  // Functor: perform backward computations
  virtual std::vector<std::vector<egr::EagerTensor>> operator()(
      const std::vector<std::vector<egr::EagerTensor>>& grads) override;

  void RetainGrad(
      const std::function<egr::EagerTensor(const egr::EagerTensor&)>& hook);

  egr::EagerTensor* Grad() { return &accumulated_grad; }

 private:
  egr::EagerTensor accumulated_grad;

  std::function<egr::EagerTensor(const egr::EagerTensor&)> retain_grad_hook_;
};

}  // namespace egr
