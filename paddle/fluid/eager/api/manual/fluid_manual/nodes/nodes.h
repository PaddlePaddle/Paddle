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
      phi::errors::InvalidArgument("Attribute(%s) cannot be found.", name));
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

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>,  // NOLINT
                           egr::kSlotSmallVectorSize>& grads,          // NOLINT
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
  void SetTensorWrapperDropout1Mask(
      const paddle::experimental::Tensor& Dropout1Mask) {
    Dropout1Mask_ = egr::TensorWrapper(Dropout1Mask, false);
  }
  void SetTensorWrapperDropout1Out(
      const paddle::experimental::Tensor& Dropout1Out) {
    Dropout1Out_ = egr::TensorWrapper(Dropout1Out, false);
  }
  void SetTensorWrapperDropout2Mask(
      const paddle::experimental::Tensor& Dropout2Mask) {
    Dropout2Mask_ = egr::TensorWrapper(Dropout2Mask, false);
  }
  void SetTensorWrapperDropout2Out(
      const paddle::experimental::Tensor& Dropout2Out) {
    auto pre_layer_norm = GetAttrWithDefault<bool>(
        attr_map_, default_attr_map_, "pre_layer_norm");
    Dropout2Out_ = egr::TensorWrapper(Dropout2Out, pre_layer_norm);
  }
  void SetTensorWrapperLinear1Bias(
      const paddle::experimental::Tensor& Linear1Bias) {
    Linear1Bias_ = egr::TensorWrapper(Linear1Bias, false);
  }
  void SetTensorWrapperLinear1Out(
      const paddle::experimental::Tensor& Linear1Out) {
    Linear1Out_ = egr::TensorWrapper(Linear1Out, false);
  }
  void SetTensorWrapperLinear1Weight(
      const paddle::experimental::Tensor& Linear1Weight) {
    Linear1Weight_ = egr::TensorWrapper(Linear1Weight, false);
  }
  void SetTensorWrapperLinear2Bias(
      const paddle::experimental::Tensor& Linear2Bias) {
    Linear2Bias_ = egr::TensorWrapper(Linear2Bias, false);
  }
  void SetTensorWrapperLinear2Weight(
      const paddle::experimental::Tensor& Linear2Weight) {
    Linear2Weight_ = egr::TensorWrapper(Linear2Weight, false);
  }
  void SetTensorWrapperLn2Bias(const paddle::experimental::Tensor& Ln2Bias) {
    Ln2Bias_ = egr::TensorWrapper(Ln2Bias, false);
  }
  void SetTensorWrapperLn2Mean(const paddle::experimental::Tensor& Ln2Mean) {
    Ln2Mean_ = egr::TensorWrapper(Ln2Mean, false);
  }
  void SetTensorWrapperLn2Scale(const paddle::experimental::Tensor& Ln2Scale) {
    Ln2Scale_ = egr::TensorWrapper(Ln2Scale, false);
  }
  void SetTensorWrapperLn2Variance(
      const paddle::experimental::Tensor& Ln2Variance) {
    Ln2Variance_ = egr::TensorWrapper(Ln2Variance, false);
  }
  void SetTensorWrapperX(const paddle::experimental::Tensor& X) {
    X_ = egr::TensorWrapper(X, false);
  }
  void SetTensorWrapperLn1Scale(const paddle::experimental::Tensor& Ln1Scale) {
    Ln1Scale_ = egr::TensorWrapper(Ln1Scale, false);
  }
  void SetTensorWrapperLn1Bias(const paddle::experimental::Tensor& Ln1Bias) {
    Ln1Bias_ = egr::TensorWrapper(Ln1Bias, false);
  }
  void SetTensorWrapperLn1Out(const paddle::experimental::Tensor& Ln1Out) {
    Ln1Out_ = egr::TensorWrapper(Ln1Out, false);
  }
  void SetTensorWrapperLn1Mean(const paddle::experimental::Tensor& Ln1Mean) {
    Ln1Mean_ = egr::TensorWrapper(Ln1Mean, false);
  }
  void SetTensorWrapperLn1Variance(
      const paddle::experimental::Tensor& Ln1Variance) {
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

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>,  // NOLINT
                           egr::kSlotSmallVectorSize>& grads,          // NOLINT
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
  void SetTensorWrapperAttnDropoutMaskOut(
      const paddle::experimental::Tensor& AttnDropoutMaskOut) {
    AttnDropoutMaskOut_ = egr::TensorWrapper(AttnDropoutMaskOut, false);
  }
  void SetTensorWrapperAttnDropoutOut(
      const paddle::experimental::Tensor& AttnDropoutOut) {
    AttnDropoutOut_ = egr::TensorWrapper(AttnDropoutOut, false);
  }
  void SetTensorWrapperBiasDropoutResidualOut(
      const paddle::experimental::Tensor& BiasDropoutResidualOut) {
    BiasDropoutResidualOut_ = egr::TensorWrapper(BiasDropoutResidualOut, false);
  }
  void SetTensorWrapperDropoutMaskOut(
      const paddle::experimental::Tensor& DropoutMaskOut) {
    DropoutMaskOut_ = egr::TensorWrapper(DropoutMaskOut, false);
  }
  void SetTensorWrapperFMHAOut(const paddle::experimental::Tensor& FMHAOut) {
    FMHAOut_ = egr::TensorWrapper(FMHAOut, false);
  }
  void SetTensorWrapperLn2Bias(const paddle::experimental::Tensor& Ln2Bias) {
    Ln2Bias_ = egr::TensorWrapper(Ln2Bias, false);
  }
  void SetTensorWrapperLn2Mean(const paddle::experimental::Tensor& Ln2Mean) {
    Ln2Mean_ = egr::TensorWrapper(Ln2Mean, false);
  }
  void SetTensorWrapperLn2Scale(const paddle::experimental::Tensor& Ln2Scale) {
    Ln2Scale_ = egr::TensorWrapper(Ln2Scale, false);
  }
  void SetTensorWrapperLn2Variance(
      const paddle::experimental::Tensor& Ln2Variance) {
    Ln2Variance_ = egr::TensorWrapper(Ln2Variance, false);
  }
  void SetTensorWrapperOutLinearBias(
      const paddle::experimental::Tensor& OutLinearBias) {
    OutLinearBias_ = egr::TensorWrapper(OutLinearBias, false);
  }
  void SetTensorWrapperOutLinearOut(
      const paddle::experimental::Tensor& OutLinearOut) {
    OutLinearOut_ = egr::TensorWrapper(OutLinearOut, true);
  }
  void SetTensorWrapperOutLinearW(
      const paddle::experimental::Tensor& OutLinearW) {
    OutLinearW_ = egr::TensorWrapper(OutLinearW, false);
  }
  void SetTensorWrapperQKOut(const paddle::experimental::Tensor& QKOut) {
    QKOut_ = egr::TensorWrapper(QKOut, true);
  }
  void SetTensorWrapperQKTVOut(const paddle::experimental::Tensor& QKTVOut) {
    QKTVOut_ = egr::TensorWrapper(QKTVOut, true);
  }
  void SetTensorWrapperQKVBias(const paddle::experimental::Tensor& QKVBias) {
    QKVBias_ = egr::TensorWrapper(QKVBias, false);
  }
  void SetTensorWrapperQKVBiasOut(
      const paddle::experimental::Tensor& QKVBiasOut) {
    QKVBiasOut_ = egr::TensorWrapper(QKVBiasOut, true);
  }
  void SetTensorWrapperQKVOut(const paddle::experimental::Tensor& QKVOut) {
    QKVOut_ = egr::TensorWrapper(QKVOut, true);
  }
  void SetTensorWrapperQKVW(const paddle::experimental::Tensor& QKVW) {
    QKVW_ = egr::TensorWrapper(QKVW, false);
  }
  void SetTensorWrapperSoftmaxOut(
      const paddle::experimental::Tensor& SoftmaxOut) {
    SoftmaxOut_ = egr::TensorWrapper(SoftmaxOut, false);
  }
  void SetTensorWrapperSrcMask(const paddle::experimental::Tensor& SrcMask) {
    SrcMask_ = egr::TensorWrapper(SrcMask, true);
  }
  void SetTensorWrapperSrcMaskOut(
      const paddle::experimental::Tensor& SrcMaskOut) {
    SrcMaskOut_ = egr::TensorWrapper(SrcMaskOut, false);
  }
  void SetTensorWrapperTransposeOut2(
      const paddle::experimental::Tensor& TransposeOut2) {
    TransposeOut2_ = egr::TensorWrapper(TransposeOut2, false);
  }
  void SetTensorWrapperX(const paddle::experimental::Tensor& X) {
    X_ = egr::TensorWrapper(X, false);
  }
  void SetTensorWrapperLnScale(const paddle::experimental::Tensor& LnScale) {
    LnScale_ = egr::TensorWrapper(LnScale, false);
  }
  void SetTensorWrapperLnBias(const paddle::experimental::Tensor& LnBias) {
    LnBias_ = egr::TensorWrapper(LnBias, false);
  }
  void SetTensorWrapperLnOut(const paddle::experimental::Tensor& LnOut) {
    LnOut_ = egr::TensorWrapper(LnOut, false);
  }
  void SetTensorWrapperLnMean(const paddle::experimental::Tensor& LnMean) {
    LnMean_ = egr::TensorWrapper(LnMean, false);
  }
  void SetTensorWrapperLnVariance(
      const paddle::experimental::Tensor& LnVariance) {
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

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>,  // NOLINT
                           egr::kSlotSmallVectorSize>& grads,          // NOLINT
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
  void SetTensorWrapperX(const paddle::experimental::Tensor& X) {
    X_ = egr::TensorWrapper(X, false);
  }
  void SetTensorWrapperY(const paddle::experimental::Tensor& Y) {
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

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>,  // NOLINT
                           egr::kSlotSmallVectorSize>& grads,          // NOLINT
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
  void SetTensorWrapperBias(const paddle::experimental::Tensor& Bias) {
    Bias_ = egr::TensorWrapper(Bias, false);
  }
  void SetTensorWrapperBiasDropoutResidualOut(
      const paddle::experimental::Tensor& BiasDropoutResidualOut) {
    BiasDropoutResidualOut_ = egr::TensorWrapper(BiasDropoutResidualOut, false);
  }
  void SetTensorWrapperDropoutMaskOut(
      const paddle::experimental::Tensor& DropoutMaskOut) {
    DropoutMaskOut_ = egr::TensorWrapper(DropoutMaskOut, false);
  }
  void SetTensorWrapperLnBias(const paddle::experimental::Tensor& LnBias) {
    LnBias_ = egr::TensorWrapper(LnBias, false);
  }
  void SetTensorWrapperLnMean(const paddle::experimental::Tensor& LnMean) {
    LnMean_ = egr::TensorWrapper(LnMean, false);
  }
  void SetTensorWrapperLnScale(const paddle::experimental::Tensor& LnScale) {
    LnScale_ = egr::TensorWrapper(LnScale, false);
  }
  void SetTensorWrapperLnVariance(
      const paddle::experimental::Tensor& LnVariance) {
    LnVariance_ = egr::TensorWrapper(LnVariance, false);
  }
  void SetTensorWrapperResidual(const paddle::experimental::Tensor& Residual) {
    Residual_ = egr::TensorWrapper(Residual, false);
  }
  void SetTensorWrapperX(const paddle::experimental::Tensor& X) {
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
