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

template <typename T>
const T& GetAttrWithDefault(
    const paddle::framework::AttributeMap& attrs,
    const paddle::framework::AttributeMap& default_attrs,
    const std::string& name) {
  auto iter1 = attrs.find(name);
  if (iter1 != attrs.end()) {
    return PADDLE_GET_CONST(T, iter1->second);
  }
  auto iter2 = default_attrs.find(name);
  if (iter2 != default_attrs.end()) {
    return PADDLE_GET_CONST(T, iter2->second);
  }
  PADDLE_THROW(
      common::errors::InvalidArgument("Attribute(%s) cannot be found.", name));
}

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

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    FMHAOut_.clear();
    GateBias_.clear();
    GateOut_.clear();
    GateWeight_.clear();
    NonbatchedBias_.clear();
    SrcMask_.clear();
    OutLinearBias_.clear();
    OutLinearWeight_.clear();
    QKVTransposeOut_.clear();
    QKVWeight_.clear();
    Query_.clear();
    SoftmaxOut_.clear();
    SoftmaxLse_.clear();
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
  void SetTensorWrapper_FMHAOut(const paddle::Tensor& FMHAOut) {
    FMHAOut_ = egr::TensorWrapper(FMHAOut, false);
  }
  void SetTensorWrapper_GateBias(const paddle::Tensor& GateBias) {
    GateBias_ = egr::TensorWrapper(GateBias, false);
  }
  void SetTensorWrapper_GateOut(const paddle::Tensor& GateOut) {
    GateOut_ = egr::TensorWrapper(GateOut, false);
  }
  void SetTensorWrapper_GateWeight(const paddle::Tensor& GateWeight) {
    GateWeight_ = egr::TensorWrapper(GateWeight, false);
  }
  void SetTensorWrapper_NonbatchedBias(const paddle::Tensor& NonbatchedBias) {
    NonbatchedBias_ = egr::TensorWrapper(NonbatchedBias, false);
  }
  void SetTensorWrapper_SrcMask(const paddle::Tensor& SrcMask) {
    SrcMask_ = egr::TensorWrapper(SrcMask, false);
  }
  void SetTensorWrapper_OutLinearBias(const paddle::Tensor& OutLinearBias) {
    OutLinearBias_ = egr::TensorWrapper(OutLinearBias, false);
  }
  void SetTensorWrapper_OutLinearWeight(const paddle::Tensor& OutLinearWeight) {
    OutLinearWeight_ = egr::TensorWrapper(OutLinearWeight, false);
  }
  void SetTensorWrapper_QKVTransposeOut(const paddle::Tensor& QKVTransposeOut) {
    QKVTransposeOut_ = egr::TensorWrapper(QKVTransposeOut, false);
  }
  void SetTensorWrapper_QKVWeight(const paddle::Tensor& QKVWeight) {
    QKVWeight_ = egr::TensorWrapper(QKVWeight, false);
  }
  void SetTensorWrapper_Query(const paddle::Tensor& Query) {
    Query_ = egr::TensorWrapper(Query, false);
  }
  void SetTensorWrapper_SoftmaxOut(const paddle::Tensor& SoftmaxOut) {
    SoftmaxOut_ = egr::TensorWrapper(SoftmaxOut, false);
  }
  void SetTensorWrapper_SoftmaxLse(const paddle::Tensor& SoftmaxLse) {
    SoftmaxLse_ = egr::TensorWrapper(SoftmaxLse, false);
  }
  void SetTensorWrapper_Key(const paddle::Tensor& Key) {
    Key_ = egr::TensorWrapper(Key, false);
  }
  void SetTensorWrapper_QueryWeight(const paddle::Tensor& QueryWeight) {
    QueryWeight_ = egr::TensorWrapper(QueryWeight, false);
  }
  void SetTensorWrapper_KeyWeight(const paddle::Tensor& KeyWeight) {
    KeyWeight_ = egr::TensorWrapper(KeyWeight, false);
  }
  void SetTensorWrapper_ValueWeight(const paddle::Tensor& ValueWeight) {
    ValueWeight_ = egr::TensorWrapper(ValueWeight, false);
  }
  void SetTensorWrapper_QueryTransposeOut(
      const paddle::Tensor& QueryTransposeOut) {
    QueryTransposeOut_ = egr::TensorWrapper(QueryTransposeOut, false);
  }
  void SetTensorWrapper_KeyTransposeOut(const paddle::Tensor& KeyTransposeOut) {
    KeyTransposeOut_ = egr::TensorWrapper(KeyTransposeOut, false);
  }
  void SetTensorWrapper_ValueTransposeOut(
      const paddle::Tensor& ValueTransposeOut) {
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
  egr::TensorWrapper SrcMask_;
  egr::TensorWrapper OutLinearBias_;
  egr::TensorWrapper OutLinearWeight_;
  egr::TensorWrapper QKVTransposeOut_;
  egr::TensorWrapper QKVWeight_;
  egr::TensorWrapper Query_;
  egr::TensorWrapper SoftmaxOut_;
  egr::TensorWrapper SoftmaxLse_;

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

class fused_feedforwardGradNodeCompat : public egr::GradNodeBase {
 public:
  fused_feedforwardGradNodeCompat() : egr::GradNodeBase() {
    VLOG(7) << " Construct fused_feedforwardGradNodeCompat ";
  }
  fused_feedforwardGradNodeCompat(size_t bwd_in_slot_num,
                                  size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(7) << " Construct fused_feedforwardGradNodeCompat ";
  }
  ~fused_feedforwardGradNodeCompat() override {
    VLOG(6) << " Destruct fused_feedforwardGradNodeCompat ";
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    Dropout1Mask_.clear();
    Dropout1Out_.clear();
    Dropout2Mask_.clear();
    Dropout2Out_.clear();
    Linear1Bias_.clear();
    Linear1Out_.clear();
    Linear1Weight_.clear();
    Linear2Bias_.clear();
    Linear2Weight_.clear();
    Ln2Bias_.clear();
    Ln2Mean_.clear();
    Ln2Scale_.clear();
    Ln2Variance_.clear();
    X_.clear();

    SetIsTensorWrappersCleared(true);
  }
  std::string name() override { return "fused_feedforwardGradNodeCompat"; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node = std::shared_ptr<fused_feedforwardGradNodeCompat>(
          new fused_feedforwardGradNodeCompat(*this));
      return copied_node;
    }
  }

  // SetX, SetY, ...
  void SetTensorWrapper_Dropout1Mask(const paddle::Tensor& Dropout1Mask) {
    Dropout1Mask_ = egr::TensorWrapper(Dropout1Mask, false);
  }
  void SetTensorWrapper_Dropout1Out(const paddle::Tensor& Dropout1Out) {
    Dropout1Out_ = egr::TensorWrapper(Dropout1Out, false);
  }
  void SetTensorWrapper_Dropout2Mask(const paddle::Tensor& Dropout2Mask) {
    Dropout2Mask_ = egr::TensorWrapper(Dropout2Mask, false);
  }
  void SetTensorWrapper_Dropout2Out(const paddle::Tensor& Dropout2Out) {
    auto pre_layer_norm = GetAttrWithDefault<bool>(
        attr_map_, default_attr_map_, "pre_layer_norm");
    Dropout2Out_ = egr::TensorWrapper(Dropout2Out, pre_layer_norm);
  }
  void SetTensorWrapper_Linear1Bias(const paddle::Tensor& Linear1Bias) {
    Linear1Bias_ = egr::TensorWrapper(Linear1Bias, false);
  }
  void SetTensorWrapper_Linear1Out(const paddle::Tensor& Linear1Out) {
    Linear1Out_ = egr::TensorWrapper(Linear1Out, false);
  }
  void SetTensorWrapper_Linear1Weight(const paddle::Tensor& Linear1Weight) {
    Linear1Weight_ = egr::TensorWrapper(Linear1Weight, false);
  }
  void SetTensorWrapper_Linear2Bias(const paddle::Tensor& Linear2Bias) {
    Linear2Bias_ = egr::TensorWrapper(Linear2Bias, false);
  }
  void SetTensorWrapper_Linear2Weight(const paddle::Tensor& Linear2Weight) {
    Linear2Weight_ = egr::TensorWrapper(Linear2Weight, false);
  }
  void SetTensorWrapper_Ln2Bias(const paddle::Tensor& Ln2Bias) {
    Ln2Bias_ = egr::TensorWrapper(Ln2Bias, false);
  }
  void SetTensorWrapper_Ln2Mean(const paddle::Tensor& Ln2Mean) {
    Ln2Mean_ = egr::TensorWrapper(Ln2Mean, false);
  }
  void SetTensorWrapper_Ln2Scale(const paddle::Tensor& Ln2Scale) {
    Ln2Scale_ = egr::TensorWrapper(Ln2Scale, false);
  }
  void SetTensorWrapper_Ln2Variance(const paddle::Tensor& Ln2Variance) {
    Ln2Variance_ = egr::TensorWrapper(Ln2Variance, false);
  }
  void SetTensorWrapper_X(const paddle::Tensor& X) {
    X_ = egr::TensorWrapper(X, false);
  }
  void SetTensorWrapper_Ln1Scale(const paddle::Tensor& Ln1Scale) {
    Ln1Scale_ = egr::TensorWrapper(Ln1Scale, false);
  }
  void SetTensorWrapper_Ln1Bias(const paddle::Tensor& Ln1Bias) {
    Ln1Bias_ = egr::TensorWrapper(Ln1Bias, false);
  }
  void SetTensorWrapper_Ln1Out(const paddle::Tensor& Ln1Out) {
    Ln1Out_ = egr::TensorWrapper(Ln1Out, false);
  }
  void SetTensorWrapper_Ln1Mean(const paddle::Tensor& Ln1Mean) {
    Ln1Mean_ = egr::TensorWrapper(Ln1Mean, false);
  }
  void SetTensorWrapper_Ln1Variance(const paddle::Tensor& Ln1Variance) {
    Ln1Variance_ = egr::TensorWrapper(Ln1Variance, false);
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
  egr::TensorWrapper Dropout1Mask_;
  egr::TensorWrapper Dropout1Out_;
  egr::TensorWrapper Dropout2Mask_;
  egr::TensorWrapper Dropout2Out_;
  egr::TensorWrapper Linear1Bias_;
  egr::TensorWrapper Linear1Out_;
  egr::TensorWrapper Linear1Weight_;
  egr::TensorWrapper Linear2Bias_;
  egr::TensorWrapper Linear2Weight_;
  egr::TensorWrapper Ln2Bias_;
  egr::TensorWrapper Ln2Mean_;
  egr::TensorWrapper Ln2Scale_;
  egr::TensorWrapper Ln2Variance_;
  egr::TensorWrapper X_;

  egr::TensorWrapper Ln1Scale_;
  egr::TensorWrapper Ln1Bias_;
  egr::TensorWrapper Ln1Out_;
  egr::TensorWrapper Ln1Mean_;
  egr::TensorWrapper Ln1Variance_;

  // Attribute Map
  paddle::framework::AttributeMap attr_map_;
  paddle::framework::AttributeMap default_attr_map_;
};

class fused_attentionGradNodeCompat : public egr::GradNodeBase {
 public:
  fused_attentionGradNodeCompat() : egr::GradNodeBase() {
    VLOG(7) << " Construct fused_attentionGradNodeCompat ";
  }
  fused_attentionGradNodeCompat(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(7) << " Construct fused_attentionGradNodeCompat ";
  }
  ~fused_attentionGradNodeCompat() override {
    VLOG(6) << " Destruct fused_attentionGradNodeCompat ";
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    AttnDropoutMaskOut_.clear();
    AttnDropoutOut_.clear();
    BiasDropoutResidualOut_.clear();
    DropoutMaskOut_.clear();
    FMHAOut_.clear();
    Ln2Bias_.clear();
    Ln2Mean_.clear();
    Ln2Scale_.clear();
    Ln2Variance_.clear();
    OutLinearBias_.clear();
    OutLinearOut_.clear();
    OutLinearW_.clear();
    QKOut_.clear();
    QKTVOut_.clear();
    QKVBias_.clear();
    QKVBiasOut_.clear();
    QKVOut_.clear();
    QKVW_.clear();
    SoftmaxOut_.clear();
    SrcMask_.clear();
    SrcMaskOut_.clear();
    TransposeOut2_.clear();
    X_.clear();

    SetIsTensorWrappersCleared(true);
  }
  std::string name() override { return "fused_attentionGradNodeCompat"; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node = std::shared_ptr<fused_attentionGradNodeCompat>(
          new fused_attentionGradNodeCompat(*this));
      return copied_node;
    }
  }

  // SetX, SetY, ...
  void SetTensorWrapper_AttnDropoutMaskOut(
      const paddle::Tensor& AttnDropoutMaskOut) {
    AttnDropoutMaskOut_ = egr::TensorWrapper(AttnDropoutMaskOut, false);
  }
  void SetTensorWrapper_AttnDropoutOut(const paddle::Tensor& AttnDropoutOut) {
    AttnDropoutOut_ = egr::TensorWrapper(AttnDropoutOut, false);
  }
  void SetTensorWrapper_BiasDropoutResidualOut(
      const paddle::Tensor& BiasDropoutResidualOut) {
    BiasDropoutResidualOut_ = egr::TensorWrapper(BiasDropoutResidualOut, false);
  }
  void SetTensorWrapper_DropoutMaskOut(const paddle::Tensor& DropoutMaskOut) {
    DropoutMaskOut_ = egr::TensorWrapper(DropoutMaskOut, false);
  }
  void SetTensorWrapper_FMHAOut(const paddle::Tensor& FMHAOut) {
    FMHAOut_ = egr::TensorWrapper(FMHAOut, false);
  }
  void SetTensorWrapper_Ln2Bias(const paddle::Tensor& Ln2Bias) {
    Ln2Bias_ = egr::TensorWrapper(Ln2Bias, false);
  }
  void SetTensorWrapper_Ln2Mean(const paddle::Tensor& Ln2Mean) {
    Ln2Mean_ = egr::TensorWrapper(Ln2Mean, false);
  }
  void SetTensorWrapper_Ln2Scale(const paddle::Tensor& Ln2Scale) {
    Ln2Scale_ = egr::TensorWrapper(Ln2Scale, false);
  }
  void SetTensorWrapper_Ln2Variance(const paddle::Tensor& Ln2Variance) {
    Ln2Variance_ = egr::TensorWrapper(Ln2Variance, false);
  }
  void SetTensorWrapper_OutLinearBias(const paddle::Tensor& OutLinearBias) {
    OutLinearBias_ = egr::TensorWrapper(OutLinearBias, false);
  }
  void SetTensorWrapper_OutLinearOut(const paddle::Tensor& OutLinearOut) {
    OutLinearOut_ = egr::TensorWrapper(OutLinearOut, true);
  }
  void SetTensorWrapper_OutLinearW(const paddle::Tensor& OutLinearW) {
    OutLinearW_ = egr::TensorWrapper(OutLinearW, false);
  }
  void SetTensorWrapper_QKOut(const paddle::Tensor& QKOut) {
    QKOut_ = egr::TensorWrapper(QKOut, true);
  }
  void SetTensorWrapper_QKTVOut(const paddle::Tensor& QKTVOut) {
    QKTVOut_ = egr::TensorWrapper(QKTVOut, true);
  }
  void SetTensorWrapper_QKVBias(const paddle::Tensor& QKVBias) {
    QKVBias_ = egr::TensorWrapper(QKVBias, false);
  }
  void SetTensorWrapper_QKVBiasOut(const paddle::Tensor& QKVBiasOut) {
    QKVBiasOut_ = egr::TensorWrapper(QKVBiasOut, true);
  }
  void SetTensorWrapper_QKVOut(const paddle::Tensor& QKVOut) {
    QKVOut_ = egr::TensorWrapper(QKVOut, true);
  }
  void SetTensorWrapper_QKVW(const paddle::Tensor& QKVW) {
    QKVW_ = egr::TensorWrapper(QKVW, false);
  }
  void SetTensorWrapper_SoftmaxOut(const paddle::Tensor& SoftmaxOut) {
    SoftmaxOut_ = egr::TensorWrapper(SoftmaxOut, false);
  }
  void SetTensorWrapper_SrcMask(const paddle::Tensor& SrcMask) {
    SrcMask_ = egr::TensorWrapper(SrcMask, true);
  }
  void SetTensorWrapper_SrcMaskOut(const paddle::Tensor& SrcMaskOut) {
    SrcMaskOut_ = egr::TensorWrapper(SrcMaskOut, false);
  }
  void SetTensorWrapper_TransposeOut2(const paddle::Tensor& TransposeOut2) {
    TransposeOut2_ = egr::TensorWrapper(TransposeOut2, false);
  }
  void SetTensorWrapper_X(const paddle::Tensor& X) {
    X_ = egr::TensorWrapper(X, false);
  }
  void SetTensorWrapper_LnScale(const paddle::Tensor& LnScale) {
    LnScale_ = egr::TensorWrapper(LnScale, false);
  }
  void SetTensorWrapper_LnBias(const paddle::Tensor& LnBias) {
    LnBias_ = egr::TensorWrapper(LnBias, false);
  }
  void SetTensorWrapper_LnOut(const paddle::Tensor& LnOut) {
    LnOut_ = egr::TensorWrapper(LnOut, false);
  }
  void SetTensorWrapper_LnMean(const paddle::Tensor& LnMean) {
    LnMean_ = egr::TensorWrapper(LnMean, false);
  }
  void SetTensorWrapper_LnVariance(const paddle::Tensor& LnVariance) {
    LnVariance_ = egr::TensorWrapper(LnVariance, false);
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
  egr::TensorWrapper AttnDropoutMaskOut_;
  egr::TensorWrapper AttnDropoutOut_;
  egr::TensorWrapper BiasDropoutResidualOut_;
  egr::TensorWrapper DropoutMaskOut_;
  egr::TensorWrapper FMHAOut_;
  egr::TensorWrapper Ln2Bias_;
  egr::TensorWrapper Ln2Mean_;
  egr::TensorWrapper Ln2Scale_;
  egr::TensorWrapper Ln2Variance_;
  egr::TensorWrapper OutLinearBias_;
  egr::TensorWrapper OutLinearOut_;
  egr::TensorWrapper OutLinearW_;
  egr::TensorWrapper QKOut_;
  egr::TensorWrapper QKTVOut_;
  egr::TensorWrapper QKVBias_;
  egr::TensorWrapper QKVBiasOut_;
  egr::TensorWrapper QKVOut_;
  egr::TensorWrapper QKVW_;
  egr::TensorWrapper SoftmaxOut_;
  egr::TensorWrapper SrcMask_;
  egr::TensorWrapper SrcMaskOut_;
  egr::TensorWrapper TransposeOut2_;
  egr::TensorWrapper X_;

  egr::TensorWrapper LnScale_;
  egr::TensorWrapper LnBias_;
  egr::TensorWrapper LnOut_;
  egr::TensorWrapper LnMean_;
  egr::TensorWrapper LnVariance_;

  // Attribute Map
  paddle::framework::AttributeMap attr_map_;
  paddle::framework::AttributeMap default_attr_map_;
};

class fused_gemm_epilogueGradNodeCompat : public egr::GradNodeBase {
 public:
  fused_gemm_epilogueGradNodeCompat() : egr::GradNodeBase() {
    VLOG(7) << " Construct fused_gemm_epilogueGradNodeCompat ";
  }
  fused_gemm_epilogueGradNodeCompat(size_t bwd_in_slot_num,
                                    size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(7) << " Construct fused_gemm_epilogueGradNodeCompat ";
  }
  ~fused_gemm_epilogueGradNodeCompat() override {
    VLOG(6) << " Destruct fused_gemm_epilogueGradNodeCompat ";
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    X_.clear();
    Y_.clear();

    SetIsTensorWrappersCleared(true);
  }
  std::string name() override { return "fused_gemm_epilogueGradNodeCompat"; }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node = std::shared_ptr<fused_gemm_epilogueGradNodeCompat>(
          new fused_gemm_epilogueGradNodeCompat(*this));
      return copied_node;
    }
  }

  // SetX, SetY, ...
  void SetTensorWrapper_X(const paddle::Tensor& X) {
    X_ = egr::TensorWrapper(X, false);
  }
  void SetTensorWrapper_Y(const paddle::Tensor& Y) {
    Y_ = egr::TensorWrapper(Y, false);
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
  egr::TensorWrapper X_;
  egr::TensorWrapper Y_;

  // Attribute Map
  paddle::framework::AttributeMap attr_map_;
  paddle::framework::AttributeMap default_attr_map_;
};

class fused_bias_dropout_residual_layer_normGradNodeCompat
    : public egr::GradNodeBase {
 public:
  fused_bias_dropout_residual_layer_normGradNodeCompat() : egr::GradNodeBase() {
    VLOG(7)
        << " Construct fused_bias_dropout_residual_layer_normGradNodeCompat ";
  }
  fused_bias_dropout_residual_layer_normGradNodeCompat(size_t bwd_in_slot_num,
                                                       size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {
    VLOG(7)
        << " Construct fused_bias_dropout_residual_layer_normGradNodeCompat ";
  }
  ~fused_bias_dropout_residual_layer_normGradNodeCompat() override {
    VLOG(6)
        << " Destruct fused_bias_dropout_residual_layer_normGradNodeCompat ";
  }

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;

  void ClearTensorWrappers() override {
    Bias_.clear();
    BiasDropoutResidualOut_.clear();
    DropoutMaskOut_.clear();
    LnBias_.clear();
    LnMean_.clear();
    LnScale_.clear();
    LnVariance_.clear();
    Residual_.clear();
    X_.clear();

    SetIsTensorWrappersCleared(true);
  }
  std::string name() override {
    return "fused_bias_dropout_residual_layer_normGradNodeCompat";
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    {
      auto copied_node =
          std::shared_ptr<fused_bias_dropout_residual_layer_normGradNodeCompat>(
              new fused_bias_dropout_residual_layer_normGradNodeCompat(*this));
      return copied_node;
    }
  }

  // SetX, SetY, ...
  void SetTensorWrapper_Bias(const paddle::Tensor& Bias) {
    Bias_ = egr::TensorWrapper(Bias, false);
  }
  void SetTensorWrapper_BiasDropoutResidualOut(
      const paddle::Tensor& BiasDropoutResidualOut) {
    BiasDropoutResidualOut_ = egr::TensorWrapper(BiasDropoutResidualOut, false);
  }
  void SetTensorWrapper_DropoutMaskOut(const paddle::Tensor& DropoutMaskOut) {
    DropoutMaskOut_ = egr::TensorWrapper(DropoutMaskOut, false);
  }
  void SetTensorWrapper_LnBias(const paddle::Tensor& LnBias) {
    LnBias_ = egr::TensorWrapper(LnBias, false);
  }
  void SetTensorWrapper_LnMean(const paddle::Tensor& LnMean) {
    LnMean_ = egr::TensorWrapper(LnMean, false);
  }
  void SetTensorWrapper_LnScale(const paddle::Tensor& LnScale) {
    LnScale_ = egr::TensorWrapper(LnScale, false);
  }
  void SetTensorWrapper_LnVariance(const paddle::Tensor& LnVariance) {
    LnVariance_ = egr::TensorWrapper(LnVariance, false);
  }
  void SetTensorWrapper_Residual(const paddle::Tensor& Residual) {
    Residual_ = egr::TensorWrapper(Residual, false);
  }
  void SetTensorWrapper_X(const paddle::Tensor& X) {
    X_ = egr::TensorWrapper(X, false);
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
  egr::TensorWrapper Bias_;
  egr::TensorWrapper BiasDropoutResidualOut_;
  egr::TensorWrapper DropoutMaskOut_;
  egr::TensorWrapper LnBias_;
  egr::TensorWrapper LnMean_;
  egr::TensorWrapper LnScale_;
  egr::TensorWrapper LnVariance_;
  egr::TensorWrapper Residual_;
  egr::TensorWrapper X_;

  // Attribute Map
  paddle::framework::AttributeMap attr_map_;
  paddle::framework::AttributeMap default_attr_map_;
};
