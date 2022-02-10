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
  virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      const std::vector<std::vector<paddle::experimental::Tensor>>& grads)
      override;

  void RetainGrad(const std::function<paddle::experimental::Tensor(
                      const paddle::experimental::Tensor&)>& hook);

  paddle::experimental::Tensor* Grad() { return &accumulated_grad; }

 private:
  paddle::experimental::Tensor accumulated_grad;

  std::function<paddle::experimental::Tensor(
      const paddle::experimental::Tensor&)>
      retain_grad_hook_;
};

}  // namespace egr
