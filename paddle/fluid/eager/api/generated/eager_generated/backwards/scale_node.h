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
#include "paddle/fluid/eager/tensor_wrapper.h"

/*
    Each Operation has a specific GradNode inheritted from GradNodeBase
    A specific GradNode defines
    1. Input Tensors
    2. overrides operator() to perform actual backward computations

    TODO: Generate GradNode via auto-code-generation
*/
namespace egr {

void ScaleAPI(const paddle::experimental::Tensor& x, float scale, float bias,
              bool bias_after_scale, paddle::experimental::Tensor* out);

class GradNodeScale : public GradNodeBase {
 public:
  // Constructor: configure fwd input tensors to grad node
  GradNodeScale(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GradNodeScale() override = default;

  // Functor: perform backward computations
  virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      std::vector<std::vector<paddle::experimental::Tensor>>& grads,  // NOLINT
      bool create_graph = false) override;

  void ClearTensorWrappers() override { VLOG(6) << "Do nothing here now"; }

  void SetTensorWrappers_X(
      const std::vector<paddle::experimental::Tensor>& tensors);

  void SetAttributes_scale(float scale);
  std::string name() override { return ""; }
  // Members: define fwd input tensors
  // For Scale there is no fwd input tensor needed

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::make_shared<GradNodeScale>(*this);
    return copied_node;
  }

 private:
  float scale_{1.0};
};

}  // namespace egr
