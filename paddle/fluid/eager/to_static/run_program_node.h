// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

class GradNodeRunProgram : public egr::GradNodeBase {
 public:
  GradNodeRunProgram(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}

  ~GradNodeRunProgram() override;

  // Functor: perform backward computations
  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize> &grads,  // NOLINT
             bool create_graph UNUSED,
             bool is_new_grad UNUSED) override;

  void ClearTensorWrappers() override {
    x_.clear();
    params_.clear();
    SetIsTensorWrappersCleared(true);
  }

  // SetAttrMap
  void SetAttrMap(const paddle::framework::AttributeMap &prog_attrs,
                  const paddle::framework::AttributeMap &cuda_graph_attrs) {
    prog_attrs_ = prog_attrs;
    cuda_graph_attrs_ = cuda_graph_attrs;
  }

  void SetFwdX(const std::vector<paddle::Tensor> &tensors) { x_ = tensors; }

  void SetFwdParams(const std::vector<paddle::Tensor> &tensors) {
    params_ = tensors;
  }

  void SetStepScope(const std::vector<paddle::framework::Scope *> &scopes) {
    step_scope_ = scopes;
  }

  void SetPlaceHashKey(const int64_t &place_hash_key) {
    place_hash_key_ = place_hash_key;
  }

 protected:
  void ConstructXGradTensors(const std::vector<paddle::Tensor> &x,
                             std::vector<paddle::Tensor> *x_grad);

  void ConstructParamGradTensors(const std::vector<paddle::Tensor> &params,
                                 std::vector<paddle::Tensor> *param_grads);

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GradNodeRunProgram>(new GradNodeRunProgram(*this));
    return copied_node;
  }

 private:
  // TensorWrappers
  std::vector<paddle::Tensor> x_;
  std::vector<paddle::Tensor> params_;
  std::vector<paddle::framework::Scope *> step_scope_;

  // Attribute Map
  paddle::framework::AttributeMap prog_attrs_;
  paddle::framework::AttributeMap cuda_graph_attrs_;

  int64_t place_hash_key_;

  std::shared_ptr<bool> executed_ = std::make_shared<bool>(false);
};

class GradNodeLegacyRunProgram : public egr::GradNodeBase {
 public:
  GradNodeLegacyRunProgram(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(4) << "GradNodeLegacyRunProgram";
  }

  ~GradNodeLegacyRunProgram() override;

  // Functor: perform backward computations
  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize> &grads,  // NOLINT
             bool create_graph UNUSED,
             bool is_new_grad UNUSED) override;

  void ClearTensorWrappers() override {
    x_.clear();
    params_.clear();
    SetIsTensorWrappersCleared(true);
  }

  // SetAttrMap
  void SetAttrMap(const paddle::framework::AttributeMap &attrs) {
    attrs_ = attrs;
  }

  void SetFwdX(const std::vector<paddle::Tensor> &tensors) { x_ = tensors; }

  void SetFwdParams(const std::vector<paddle::Tensor> &tensors) {
    params_ = tensors;
  }

  void SetStepScope(const std::vector<paddle::framework::Scope *> &scopes) {
    step_scope_ = scopes;
  }

  void SetPlaceHashKey(const int64_t &place_hash_key) {
    place_hash_key_ = place_hash_key;
  }

 protected:
  void ConstructXGradTensors(const std::vector<paddle::Tensor> &x,
                             std::vector<paddle::Tensor> *x_grad);

  void ConstructParamGradTensors(const std::vector<paddle::Tensor> &params,
                                 std::vector<paddle::Tensor> *param_grads);

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<GradNodeLegacyRunProgram>(
        new GradNodeLegacyRunProgram(*this));
    return copied_node;
  }

 private:
  // TensorWrappers
  std::vector<paddle::Tensor> x_;
  std::vector<paddle::Tensor> params_;
  std::vector<paddle::framework::Scope *> step_scope_;

  // Attribute Map
  paddle::framework::AttributeMap attrs_;

  int64_t place_hash_key_;

  // why use shared_ptr. because paddle.grad will copy GradNode, if
  // we use bool, the copied node have different executed states.
  std::shared_ptr<bool> executed_ = std::make_shared<bool>(false);
};
