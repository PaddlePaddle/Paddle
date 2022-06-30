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

#pragma once
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/imperative/tracer.h"

class fused_gate_attentionGradNodeCompat : public egr::GradNodeBase {
 public:
  fused_gate_attentionGradNodeCompat() : egr::GradNodeBase() {
    VLOG(7) << " Construct fused_gate_attentionGradNodeCompat ";
  }
  fused_gate_attentionGradNodeCompat(size_t bwd_in_slot_num,
                                     size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(7) << " Construct fused_gate_attentionGradNodeCompat ";
  }
  ~fused_gate_attentionGradNodeCompat() override {
    VLOG(6) << " Destruct fused_gate_attentionGradNodeCompat ";
  }

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>,  // NOLINT
                           egr::kSlotSmallVectorSize>& grads,          // NOLINT
      bool create_graph = false,
      bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    FMHAOut_.clear();
    GateBias_.clear();
    GateOut_.clear();
    GateWeight_.clear();
    NonbatchedBias_.clear();
    OutLinearBias_.clear();
    OutLinearWeight_.clear();
    QKVTransposeOut_.clear();
    QKVWeight_.clear();
    Query_.clear();
    SoftmaxOut_.clear();
    Key_.clear();
    QueryWeight_.clear();
    KeyWeight_.clear();
    ValueWeight_.clear();
    QueryTransposeOut_.clear();
    KeyTransposeOut_.clear();
    ValueTransposeOut_.clear();

    SetIsTensorWrappersCleared(true);
  }
  std::string name() override { return "fused_gate_attentionGradNodeCompat"; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node = std::shared_ptr<fused_gate_attentionGradNodeCompat>(
          new fused_gate_attentionGradNodeCompat(*this));
      return copied_node;
    }
  }

  // SetX, SetY, ...
  void SetTensorWrapperFMHAOut(const paddle::experimental::Tensor& FMHAOut) {
    FMHAOut_ = egr::TensorWrapper(FMHAOut, false);
  }
  void SetTensorWrapperGateBias(const paddle::experimental::Tensor& GateBias) {
    GateBias_ = egr::TensorWrapper(GateBias, false);
  }
  void SetTensorWrapperGateOut(const paddle::experimental::Tensor& GateOut) {
    GateOut_ = egr::TensorWrapper(GateOut, false);
  }
  void SetTensorWrapperGateWeight(
      const paddle::experimental::Tensor& GateWeight) {
    GateWeight_ = egr::TensorWrapper(GateWeight, false);
  }
  void SetTensorWrapperNonbatchedBias(
      const paddle::experimental::Tensor& NonbatchedBias) {
    NonbatchedBias_ = egr::TensorWrapper(NonbatchedBias, false);
  }
  void SetTensorWrapperOutLinearBias(
      const paddle::experimental::Tensor& OutLinearBias) {
    OutLinearBias_ = egr::TensorWrapper(OutLinearBias, false);
  }
  void SetTensorWrapperOutLinearWeight(
      const paddle::experimental::Tensor& OutLinearWeight) {
    OutLinearWeight_ = egr::TensorWrapper(OutLinearWeight, false);
  }
  void SetTensorWrapperQKVTransposeOut(
      const paddle::experimental::Tensor& QKVTransposeOut) {
    QKVTransposeOut_ = egr::TensorWrapper(QKVTransposeOut, false);
  }
  void SetTensorWrapperQKVWeight(
      const paddle::experimental::Tensor& QKVWeight) {
    QKVWeight_ = egr::TensorWrapper(QKVWeight, false);
  }
  void SetTensorWrapperQuery(const paddle::experimental::Tensor& Query) {
    Query_ = egr::TensorWrapper(Query, false);
  }
  void SetTensorWrapperSoftmaxOut(
      const paddle::experimental::Tensor& SoftmaxOut) {
    SoftmaxOut_ = egr::TensorWrapper(SoftmaxOut, false);
  }
  void SetTensorWrapperKey(const paddle::experimental::Tensor& Key) {
    Key_ = egr::TensorWrapper(Key, false);
  }
  void SetTensorWrapperQueryWeight(
      const paddle::experimental::Tensor& QueryWeight) {
    QueryWeight_ = egr::TensorWrapper(QueryWeight, false);
  }
  void SetTensorWrapperKeyWeight(
      const paddle::experimental::Tensor& KeyWeight) {
    KeyWeight_ = egr::TensorWrapper(KeyWeight, false);
  }
  void SetTensorWrapperValueWeight(
      const paddle::experimental::Tensor& ValueWeight) {
    ValueWeight_ = egr::TensorWrapper(ValueWeight, false);
  }
  void SetTensorWrapperQueryTransposeOut(
      const paddle::experimental::Tensor& QueryTransposeOut) {
    QueryTransposeOut_ = egr::TensorWrapper(QueryTransposeOut, false);
  }
  void SetTensorWrapperKeyTransposeOut(
      const paddle::experimental::Tensor& KeyTransposeOut) {
    KeyTransposeOut_ = egr::TensorWrapper(KeyTransposeOut, false);
  }
  void SetTensorWrapperValueTransposeOut(
      const paddle::experimental::Tensor& ValueTransposeOut) {
    ValueTransposeOut_ = egr::TensorWrapper(ValueTransposeOut, false);
  }

  // SetAttrMap
  void SetAttrMap(paddle::framework::AttributeMap&& attr_map) {
    attr_map_ = std::move(attr_map);
  }
  void SetDefaultAttrMap(paddle::framework::AttributeMap&& default_attr_map) {
    default_attr_map_ = std::move(default_attr_map);
  }

 private:
  // TensorWrappers
  egr::TensorWrapper FMHAOut_;
  egr::TensorWrapper GateBias_;
  egr::TensorWrapper GateOut_;
  egr::TensorWrapper GateWeight_;
  egr::TensorWrapper NonbatchedBias_;
  egr::TensorWrapper OutLinearBias_;
  egr::TensorWrapper OutLinearWeight_;
  egr::TensorWrapper QKVTransposeOut_;
  egr::TensorWrapper QKVWeight_;
  egr::TensorWrapper Query_;
  egr::TensorWrapper SoftmaxOut_;

  egr::TensorWrapper Key_;
  egr::TensorWrapper QueryWeight_;
  egr::TensorWrapper KeyWeight_;
  egr::TensorWrapper ValueWeight_;
  egr::TensorWrapper QueryTransposeOut_;
  egr::TensorWrapper KeyTransposeOut_;
  egr::TensorWrapper ValueTransposeOut_;

  // Attribute Map
  paddle::framework::AttributeMap attr_map_;
  paddle::framework::AttributeMap default_attr_map_;
};
