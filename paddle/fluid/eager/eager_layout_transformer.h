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

#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
namespace egr {
inline paddle::Tensor EagerTraceTransposeOp(const phi::DataLayout layout,
                                            const paddle::Tensor& in) {
  VLOG(4) << "AutoTune Transpose from " << in.layout() << " to " << layout
          << ", tensor's dim size is " << in.shape().size();
  if (in.shape().size() != 4) {
    return in;
  }
  std::vector<int> axis;
  if (layout == phi::DataLayout::NHWC) {
    axis = {0, 2, 3, 1};
  } else if (layout == phi::DataLayout::NCHW) {
    axis = {0, 3, 1, 2};
  } else {
    axis = {0, 1, 2, 3};
  }
  auto out_tensor = trans_layout_ad_func(in, axis);
  VLOG(4) << "AutoTune Transpose from " << in.layout() << " to " << layout;
  return out_tensor;
}

inline phi::DataLayout DesiredLayout() {
  return paddle::imperative::LayoutAutoTune::Instance().GetDesiredLayout();
}

inline phi::DataLayout DefaultLayout() {
  return paddle::imperative::LayoutAutoTune::Instance().GetDefaultLayout();
}

inline void UpdateLayout(paddle::Tensor* out_tensor,
                         const phi::DataLayout layout) {
  if (out_tensor->layout() != layout) {
    VLOG(4) << "Update out_tensor's layout from " << out_tensor->layout()
            << " to " << layout;
    phi::DenseTensorUtils::GetMutableMeta(
        static_cast<phi::DenseTensor*>(out_tensor->impl().get()))
        ->layout = layout;
  }
}

inline void DealWithShapeOp(paddle::Tensor* out_tensor,
                            const phi::DataLayout layout,
                            int dim_size) {
  auto des_layout = DesiredLayout();
  auto def_layout = DefaultLayout();
  int32_t* value =
      static_cast<phi::DenseTensor*>(out_tensor->impl().get())->data<int32_t>();
  bool change_dim =
      (des_layout != def_layout && layout == des_layout && dim_size == 4);
  VLOG(6) << "'Shape OP', layout autotune: True"
          << " desired_layout: " << des_layout
          << " default_layout: " << def_layout
          << " tensor layout: " << out_tensor->layout()
          << " tensor's shape size is : " << dim_size;
  // It's means input tensor has been autotune and tensor's layout is
  // desired_layout
  std::vector<int32_t> dims;
  dims.resize(dim_size);
  for (int i = 0; i < dim_size; i++) {
    dims[i] = value[i];
  }
  auto des_str = common::DataLayoutToString(des_layout);
  if (change_dim && des_str == "NCHW") {
    // NCHW -> NHWC
    VLOG(6) << "layout autotune get Shape from NCHW -> NHWC " << value[0] << " "
            << value[1] << " " << value[2] << " " << value[3] << " to "
            << dims[0] << " " << dims[2] << " " << dims[3] << " " << dims[1];
    value[0] = dims[0];
    value[1] = dims[2];
    value[2] = dims[3];
    value[3] = dims[1];
  } else if (change_dim && des_str == "NHWC") {
    // NHWC -> NCHW
    VLOG(6) << "layout autotune get Shape from NHWC -> NCHW " << value[0] << " "
            << value[1] << " " << value[2] << " " << value[3] << " to "
            << dims[0] << " " << dims[3] << " " << dims[1] << " " << dims[2];
    value[0] = dims[0];
    value[1] = dims[3];
    value[2] = dims[1];
    value[3] = dims[2];
  }
}

// agnostic op
class EagerLayoutTransformer {
  using Layout = phi::DataLayout;

 public:
  EagerLayoutTransformer() : op_name_(""), final_layout_(Layout::UNDEFINED) {}

  EagerLayoutTransformer(const EagerLayoutTransformer&) = delete;

  EagerLayoutTransformer& operator=(const EagerLayoutTransformer&) = delete;

  explicit EagerLayoutTransformer(
      const std::string& op_name,
      const paddle::small_vector<std::vector<paddle::Tensor>,
                                 kSlotSmallVectorSize>& tensors_vector UNUSED,
      const Layout final_layout = Layout::UNDEFINED)
      : op_name_(op_name), final_layout_(final_layout), dim_size_(1) {
    VLOG(4) << "Agnostic op : " << op_name_ << "'s layout is " << final_layout_;
  }

  virtual ~EagerLayoutTransformer() {}

  virtual paddle::Tensor TransInTensor(const std::string& in_name UNUSED,
                                       const paddle::Tensor& in) {
    // update in shape size
    dim_size_ = in.shape().size();
    bool need_trans =
        !(final_layout_ == Layout::UNDEFINED || final_layout_ == in.layout());
    // This is for Agnostic op when layout is different
    if (need_trans) {
      auto out_tensor = EagerTraceTransposeOp(final_layout_, in);
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor.impl().get()))
          ->layout = final_layout_;
      return out_tensor;
    }
    return in;
  }

  virtual paddle::optional<paddle::Tensor> TransInTensor(
      const std::string& in_name, const paddle::optional<paddle::Tensor>& in) {
    return in ? TransInTensor(in_name, *in) : in;
  }

  virtual std::vector<paddle::Tensor> TransInTensors(
      const std::string& in_name UNUSED,
      const std::vector<paddle::Tensor>& in) {
    return in;
  }

  virtual paddle::optional<std::vector<paddle::Tensor>> TransInTensors(
      const std::string& in_name,
      const paddle::optional<std::vector<paddle::Tensor>>& in) {
    return (in ? TransInTensors(in_name, *in) : in);
  }

  virtual void SetOutTensorLayout(std::vector<paddle::Tensor>* out_tensor) {
    bool update_layout = !(final_layout_ == Layout::UNDEFINED);
    if (update_layout) {
      for (size_t i = 0; i < out_tensor->size(); i++) {
        phi::DenseTensorUtils::GetMutableMeta(
            static_cast<phi::DenseTensor*>((*out_tensor)[i].impl().get()))
            ->layout = DesiredLayout();
      }
    }
  }

  virtual void SetOutTensorLayout(
      paddle::optional<paddle::Tensor>* out_tensor UNUSED) {
    VLOG(4) << "AutoTune out tensor is optional";
  }

  virtual void SetOutTensorLayout(
      paddle::optional<std::vector<paddle::Tensor>>* out_tensor UNUSED) {
    VLOG(4) << "AutoTune out tensor is optional";
  }

  virtual void SetOutTensorLayout(paddle::Tensor* out_tensor) {
    if (op_name_ == "shape") {
      return DealWithShapeOp(out_tensor, final_layout_, dim_size_);
    }
    bool need_update = !(final_layout_ == Layout::UNDEFINED);
    if (need_update) {
      UpdateLayout(out_tensor, final_layout_);
    }
  }

 protected:
  std::string op_name_;
  const Layout final_layout_;
  int dim_size_;
};

class EagerHeavilyLayoutSensitiveOpTransformer : public EagerLayoutTransformer {
 public:
  explicit EagerHeavilyLayoutSensitiveOpTransformer(const std::string& op_name,
                                                    std::string* layout)
      : op_name_(op_name), desired_layout_(DesiredLayout()) {
    VLOG(4) << "Heavily op: " << op_name << " layout " << *layout;
    *layout = common::DataLayoutToString(DesiredLayout());
  }

  paddle::Tensor TransInTensor(const std::string& in_name,
                               const paddle::Tensor& in) override {
    if (heavily_input_.count(in_name) != 0 && in.layout() != desired_layout_) {
      auto out_tensor = EagerTraceTransposeOp(desired_layout_, in);
      return out_tensor;
    }
    return in;
  }

  using EagerLayoutTransformer::SetOutTensorLayout;
  void SetOutTensorLayout(paddle::Tensor* out_tensor) override {
    UpdateLayout(out_tensor, desired_layout_);
  }

  void SetOutTensorLayout(std::vector<paddle::Tensor*>* out_tensor) {
    for (size_t i = 0; i < out_tensor->size(); i++) {
      SetOutTensorLayout((*out_tensor)[i]);
    }
  }

  void SetOutTensorLayout(std::vector<paddle::Tensor>* out_tensor) override {
    for (size_t i = 0; i < out_tensor->size(); i++) {
      if ((*out_tensor)[i].layout() != desired_layout_) {
        VLOG(4) << "Update out_tensor's layout from "
                << (*out_tensor)[i].layout() << " to " << desired_layout_;
        phi::DenseTensorUtils::GetMutableMeta(
            static_cast<phi::DenseTensor*>((*out_tensor)[i].impl().get()))
            ->layout = desired_layout_;
      }
    }
  }

 protected:
  std::string op_name_;
  const phi::DataLayout desired_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

class EagerLightlyLayoutSensitiveOpTransformer : public EagerLayoutTransformer {
 public:
  EagerLightlyLayoutSensitiveOpTransformer() {}
  explicit EagerLightlyLayoutSensitiveOpTransformer(
      const std::string& op_name) {
    VLOG(4) << "Lightly op : " << op_name;
    auto desired_layout = DesiredLayout();
    final_layout_ = common::DataLayoutToString(desired_layout);
  }

  // transpose from desired to default
  paddle::Tensor TransInTensor(const std::string& in_name UNUSED,
                               const paddle::Tensor& in) override {
    std::string input_layout = common::DataLayoutToString(in.layout());
    auto default_layout = DefaultLayout();
    if (final_layout_ == input_layout && in.shape().size() == 4) {
      auto out_tensor = EagerTraceTransposeOp(phi::DataLayout::UNDEFINED, in);
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>(out_tensor.impl().get()))
          ->layout = default_layout;
      return out_tensor;
    }
    return in;
  }

  std::vector<paddle::Tensor> TransInTensors(
      const std::string& in_name UNUSED,
      const std::vector<paddle::Tensor>& in) override {
    std::vector<paddle::Tensor> result;
    auto desired_layout = DesiredLayout();
    auto default_layout = DefaultLayout();
    for (size_t i = 0; i < in.size(); i++) {
      auto in_tensor = in[i];
      if (in_tensor.layout() == desired_layout) {
        auto out_tensor =
            EagerTraceTransposeOp(phi::DataLayout::UNDEFINED, in_tensor);
        phi::DenseTensorUtils::GetMutableMeta(
            static_cast<phi::DenseTensor*>(out_tensor.impl().get()))
            ->layout = default_layout;
        result.emplace_back(out_tensor);
      } else {
        result.emplace_back(in_tensor);
      }
    }
    return result;
  }

  using EagerLayoutTransformer::SetOutTensorLayout;
  void SetOutTensorLayout(paddle::Tensor* out_tensor) override {
    UpdateLayout(out_tensor, DefaultLayout());
  }

  void SetOutTensorLayout(std::vector<paddle::Tensor*>* out_tensor) {
    for (size_t i = 0; i < out_tensor->size(); i++) {
      SetOutTensorLayout((*out_tensor)[i]);
    }
  }

  void SetOutTensorLayout(std::vector<paddle::Tensor>* out_tensor) override {
    auto default_layout = DefaultLayout();
    for (size_t i = 0; i < out_tensor->size(); i++) {
      phi::DenseTensorUtils::GetMutableMeta(
          static_cast<phi::DenseTensor*>((*out_tensor)[i].impl().get()))
          ->layout = default_layout;
    }
  }

 protected:
  std::string final_layout_;
  std::unordered_set<std::string> heavily_input_{"x", "y", "input"};
};

class EagerTransposeOpTransformer
    : public EagerLightlyLayoutSensitiveOpTransformer {
 public:
  EagerTransposeOpTransformer() {}
  explicit EagerTransposeOpTransformer(const std::string& op_name) {
    VLOG(4) << "AutoTuneTransformer op: " << op_name;
  }

  void SetAttr(std::vector<int>* axis, bool is_nhwc) {
    std::vector<int> perm_nchw = {0, 2, 3, 1};
    std::vector<int> perm_nhwc = {0, 3, 1, 2};
    auto perm = is_nhwc ? perm_nhwc : perm_nchw;
    (*axis)[0] = perm[(*axis)[0]];
    (*axis)[1] = perm[(*axis)[1]];
    (*axis)[2] = perm[(*axis)[2]];
    (*axis)[3] = perm[(*axis)[3]];
  }

  paddle::Tensor TransInTensor(const std::string& in_name UNUSED,
                               const paddle::Tensor& in) override {
    return in;
  }

  void SetOutTensorLayout(paddle::Tensor* out_tensor) override {
    UpdateLayout(out_tensor, DefaultLayout());
  }
};

class EagerArgmaxOpTransformer
    : public EagerLightlyLayoutSensitiveOpTransformer {
 public:
  EagerArgmaxOpTransformer() {}
  explicit EagerArgmaxOpTransformer(const std::string& op_name) {
    VLOG(4) << "AutoTuneTransformer op: " << op_name;
  }

  void SetAttr(paddle::experimental::Scalar* axis, bool is_nhwc) {
    std::vector<int> perm_nhwc = {0, 3, 1, 2};
    std::vector<int> perm_nchw = {0, 2, 3, 1};
    auto perm = is_nhwc ? perm_nhwc : perm_nchw;
    int axes = axis->to<int>();
    (*axis) = static_cast<paddle::experimental::Scalar>(perm[axes]);
  }

  void SetOutTensorLayout(paddle::Tensor* out_tensor) override {
    UpdateLayout(out_tensor, DesiredLayout());
  }
};

class EagerFlattenOpTransformer
    : public EagerLightlyLayoutSensitiveOpTransformer {
 public:
  EagerFlattenOpTransformer() {}
  explicit EagerFlattenOpTransformer(const std::string& op_name) {
    VLOG(4) << "AutoTuneTransformer op: " << op_name;
  }

  // transpose from NHWC to NCHW
  paddle::Tensor TransInTensor(const std::string& in_name UNUSED,
                               const paddle::Tensor& in) override {
    return in;
  }

  void SetOutTensorLayout(paddle::Tensor* out_tensor) override {
    UpdateLayout(out_tensor, DefaultLayout());
  }
};

class EagerConcatOpTransformer
    : public EagerLightlyLayoutSensitiveOpTransformer {
 public:
  EagerConcatOpTransformer() {}
  explicit EagerConcatOpTransformer(const std::string& op_name) {
    VLOG(4) << "AutoTuneTransformer op : " << op_name;
  }

  void SetAttr(paddle::experimental::Scalar* axis, phi::DataLayout layout) {
    std::vector<int> perm_nhwc = {0, 3, 1, 2};
    std::vector<int> perm_nchw = {0, 2, 3, 1};
    int axes = axis->to<int>();
    axes = axes < 0 ? axes + 4 : axes;
    auto perm = (phi::DataLayout::NHWC == layout) ? perm_nhwc : perm_nchw;
    (*axis) = static_cast<paddle::experimental::Scalar>(perm[axes]);
  }

  std::vector<paddle::Tensor> TransInTensors(
      const std::string& in_name UNUSED,
      const std::vector<paddle::Tensor>& in) override {
    return in;
  }

  void SetOutTensorLayout(paddle::Tensor* out_tensor) override {
    UpdateLayout(out_tensor, DesiredLayout());
  }
};
}  // namespace egr
