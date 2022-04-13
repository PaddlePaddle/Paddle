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

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/utils/any.h"

namespace egr {
class RunCustomOpNode : public GradNodeBase {
 public:
  // Constructor: configure fwd input tensors to grad node
  explicit RunCustomOpNode(size_t bwd_in_slot_num, size_t bwd_out_slot_num,
                           const std::string& op_type)
      : GradNodeBase(bwd_in_slot_num, bwd_out_slot_num), op_type_(op_type) {
    VLOG(6) << "Construct RunCustomOpNode for op: " << op_type;
  }

  ~RunCustomOpNode() override {
    VLOG(6) << "Destruct RunCustomOpNode for op: " << op_type_;
  }

  // Functor: perform backward computations
  virtual std::vector<std::vector<paddle::experimental::Tensor>>
  operator()(                                                         // NOLINT
      std::vector<std::vector<paddle::experimental::Tensor>>& grads,  // NOLINT
      bool create_graph = false)                                      // NOLINT
      override;

  std::string name() {
    return paddle::string::Sprintf("RunCustomOpNode: %s_grad", op_type_);
  }

  static std::vector<egr::TensorWrapper> ConstructTensorWrapper(
      const std::vector<paddle::experimental::Tensor>& fwd_var) {
    std::vector<egr::TensorWrapper> res;
    for (auto const& var : fwd_var) {
      res.emplace_back(var);
    }
    return res;
  }

  static std::vector<paddle::experimental::Tensor> Recover(
      std::vector<egr::TensorWrapper>* fwd_var) {
    std::vector<paddle::experimental::Tensor> res;
    for (size_t i = 0; i < fwd_var->size(); i++) {
      res.emplace_back(fwd_var->at(i).recover());
    }
    return res;
  }

  void ClearTensorWrappers() override { VLOG(6) << "Do nothing here now"; }

  void SetAttrs(const std::vector<paddle::any>& attr) { attrs_ = attr; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<RunCustomOpNode>(new RunCustomOpNode(*this));
    return copied_node;
  }

 public:
  std::unordered_map<int, std::vector<egr::TensorWrapper>> fwd_outs;
  std::unordered_map<int, std::vector<egr::TensorWrapper>> fwd_ins;
  std::unordered_map<int, int> grads2grad_in_map;

 private:
  std::vector<paddle::any> attrs_;
  std::string op_type_{""};
};

}  // namespace egr
