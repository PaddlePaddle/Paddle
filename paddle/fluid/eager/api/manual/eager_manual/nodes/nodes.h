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

class Conv2dGradNodeFinal : public egr::GradNodeBase {
 public:
  Conv2dGradNodeFinal() : egr::GradNodeBase() {}
  Conv2dGradNodeFinal(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv2dGradNodeFinal() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,                               // NOLINT
             bool is_new_grad = false) override;                      // NOLINT
  std::string name() override { return "Conv2dGradNodeFinal"; }

  void ClearTensorWrappers() override {
    input_.clear();
    filter_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<Conv2dGradNodeFinal>(new Conv2dGradNodeFinal(*this));
    VLOG(3) << "Copy Conv2dGradNodeFinal: " << this
            << " to: " << copied_node.get();
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperfilter(const paddle::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class Conv2dDoubleGradNodeFinal : public egr::GradNodeBase {
 public:
  Conv2dDoubleGradNodeFinal() : egr::GradNodeBase() {}
  Conv2dDoubleGradNodeFinal(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv2dDoubleGradNodeFinal() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,                               // NOLINT
             bool is_new_grad = false) override;                      // NOLINT
  std::string name() override { return "Conv2dDoubleGradNodeFinal"; }

  void ClearTensorWrappers() override {
    input_.clear();
    filter_.clear();
    grad_out_.clear();

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node = std::shared_ptr<Conv2dDoubleGradNodeFinal>(
        new Conv2dDoubleGradNodeFinal(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperinput(const paddle::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperfilter(const paddle::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }
  void SetTensorWrappergrad_out(const paddle::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepadding_algorithm(const std::string& padding_algorithm) {
    padding_algorithm_ = padding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;
  egr::TensorWrapper grad_out_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string padding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
};

class AddNGradNodeFinal : public egr::GradNodeBase {
 public:
  AddNGradNodeFinal() : egr::GradNodeBase() {}
  AddNGradNodeFinal(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~AddNGradNodeFinal() override = default;

  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,        // NOLINT
                                  egr::kSlotSmallVectorSize>& grads,  // NOLINT
             bool create_graph = false,
             bool is_new_grad = false) override;
  std::string name() override { return "AddNGradNodeFinal"; }

  void ClearTensorWrappers() override {
    for (auto& tw : x_) {
      tw.clear();
    }

    SetIsTensorWrappersCleared(true);
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<AddNGradNodeFinal>(new AddNGradNodeFinal(*this));
    return copied_node;
  }

  // SetTensorWrapperX, SetTensorWrapperY, ...
  void SetTensorWrapperx(const std::vector<paddle::Tensor>& x) {
    for (const auto& eager_tensor : x) {
      x_.emplace_back(egr::TensorWrapper(eager_tensor, true));
    }
  }

  // SetAttributes

 private:
  // TensorWrappers
  std::vector<egr::TensorWrapper> x_;

  // Attributes
};
