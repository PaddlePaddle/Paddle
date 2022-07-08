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

class Conv2dGradNodeFinal : public egr::GradNodeBase {
 public:
  Conv2dGradNodeFinal() : egr::GradNodeBase() {}
  Conv2dGradNodeFinal(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv2dGradNodeFinal() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>,  // NOLINT
                           egr::kSlotSmallVectorSize>& grads,          // NOLINT
      bool create_graph = false,                                       // NOLINT
      bool is_new_grad = false) override;                              // NOLINT
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
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepaddding_algorithm(const std::string& paddding_algorithm) {
    paddding_algorithm_ = paddding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }
  void SetAttributeuse_addto(const bool& use_addto) { use_addto_ = use_addto; }
  void SetAttributeworkspace_size_MB(const int& workspace_size_MB) {
    workspace_size_MB_ = workspace_size_MB;
  }
  void SetAttributeexhaustive_search(const bool& exhaustive_search) {
    exhaustive_search_ = exhaustive_search;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string paddding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
  bool use_addto_;
  int workspace_size_MB_;
  bool exhaustive_search_;
};

class Conv2dDoubleGradNodeFinal : public egr::GradNodeBase {
 public:
  Conv2dDoubleGradNodeFinal() : egr::GradNodeBase() {}
  Conv2dDoubleGradNodeFinal(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~Conv2dDoubleGradNodeFinal() override = default;

  virtual paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(
      paddle::small_vector<std::vector<paddle::experimental::Tensor>,  // NOLINT
                           egr::kSlotSmallVectorSize>& grads,          // NOLINT
      bool create_graph = false,                                       // NOLINT
      bool is_new_grad = false) override;                              // NOLINT
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
  void SetTensorWrapperinput(const paddle::experimental::Tensor& input) {
    input_ = egr::TensorWrapper(input, false);
  }
  void SetTensorWrapperfilter(const paddle::experimental::Tensor& filter) {
    filter_ = egr::TensorWrapper(filter, false);
  }
  void SetTensorWrappergrad_out(const paddle::experimental::Tensor& grad_out) {
    grad_out_ = egr::TensorWrapper(grad_out, false);
  }

  // SetAttributes
  void SetAttributestrides(const std::vector<int>& strides) {
    strides_ = strides;
  }
  void SetAttributepaddings(const std::vector<int>& paddings) {
    paddings_ = paddings;
  }
  void SetAttributepaddding_algorithm(const std::string& paddding_algorithm) {
    paddding_algorithm_ = paddding_algorithm;
  }
  void SetAttributegroups(const int& groups) { groups_ = groups; }
  void SetAttributedilations(const std::vector<int>& dilations) {
    dilations_ = dilations;
  }
  void SetAttributedata_format(const std::string& data_format) {
    data_format_ = data_format;
  }
  void SetAttributeuse_addto(const bool& use_addto) { use_addto_ = use_addto; }
  void SetAttributeworkspace_size_MB(const int& workspace_size_MB) {
    workspace_size_MB_ = workspace_size_MB;
  }
  void SetAttributeexhaustive_search(const bool& exhaustive_search) {
    exhaustive_search_ = exhaustive_search;
  }

 private:
  // TensorWrappers
  egr::TensorWrapper input_;
  egr::TensorWrapper filter_;
  egr::TensorWrapper grad_out_;

  // Attributes
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::string paddding_algorithm_;
  int groups_;
  std::vector<int> dilations_;
  std::string data_format_;
  bool use_addto_;
  int workspace_size_MB_;
  bool exhaustive_search_;
};
