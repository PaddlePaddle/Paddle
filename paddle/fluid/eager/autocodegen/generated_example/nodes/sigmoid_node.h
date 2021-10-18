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

#include "paddle/fluid/eager/grad_node_info.h"

class GradNodesigmoid : public egr::GradNodeBase {
 public:
  GradNodesigmoid() : egr::GradNodeBase() {}
  GradNodesigmoid(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GradNodesigmoid() override = default;

  virtual std::vector<std::vector<paddle::experimental::Tensor>> operator()(
      const std::vector<std::vector<paddle::experimental::Tensor>>& grads)
      override;

  // SetX, SetY, ...
  void SetTensorWrapperOut(const paddle::experimental::Tensor& Out) {
    Out_ = Out;
  }

  // SetAttr0, SetAttr1, ...
  void SetAttruse_cudnn(const bool use_cudnn) { use_cudnn_ = use_cudnn; }
  void SetAttruse_mkldnn(const bool use_mkldnn) { use_mkldnn_ = use_mkldnn; }

 private:
  // TensorWrappers
  paddle::experimental::Tensor Out_;

  // Attribute Members
  bool use_cudnn_ = 0;
  bool use_mkldnn_ = 0;
};
