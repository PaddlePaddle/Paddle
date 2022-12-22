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

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/attribute.h"

namespace paddle {
namespace operators {

// This file is to be compatible with the bad design and
// implementation of fluid in the past

// Many operators in fluid have extra attributes, which are generally added
// to implement some specific kernel selection and to meet the specialization
// needs of a specific operation library like mkldnn or cudnn
enum class ExtraAttrProperty : uint8_t {
  // The attributes that are no longer used by any scene
  DEPRECATED = 0,
  // The attributes used for framework execution scheduling,
  // such as `use_mkldnn`, `use_cudnn`, no need to save
  SCHEDULE,
  // The attributes for ONEDNN only, can be saved in OneDNNContext
  ONEDNN,
  // The attributes for GPUDNN only, can be saved in GPUContext
  GPUDNN,
  // Add necessary properties as needed
};

class ExtraAttrPropertySet final {
 public:
  constexpr ExtraAttrPropertySet() : bitset_(0) {}
  constexpr ExtraAttrPropertySet(ExtraAttrProperty e)  // NOLINT
      : bitset_(e == ExtraAttrProperty::DEPRECATED
                    ? 0
                    : 1ULL << (static_cast<uint8_t>(e) - 1)) {}

  inline uint64_t bitset() const { return bitset_; }

  bool inline Support(ExtraAttrProperty e) const {
    // DEPRECATED ExtraAttr always return false
    return static_cast<bool>(bitset_ & ExtraAttrPropertySet(e).bitset());
  }
  bool IsEmpty() const { return bitset_ == 0; }

  ExtraAttrPropertySet operator|(const ExtraAttrPropertySet& other) const {
    return ExtraAttrPropertySet(bitset_ | other.bitset());
  }
  ExtraAttrPropertySet operator&(const ExtraAttrPropertySet& other) const {
    return ExtraAttrPropertySet(bitset_ & other.bitset());
  }
  ExtraAttrPropertySet operator-(const ExtraAttrPropertySet& other) const {
    return ExtraAttrPropertySet(bitset_ & ~other.bitset());
  }
  ExtraAttrPropertySet operator^(const ExtraAttrPropertySet& other) const {
    return ExtraAttrPropertySet(bitset_ ^ other.bitset());
  }

  bool operator==(const ExtraAttrPropertySet& other) const {
    return bitset_ == other.bitset();
  }

 private:
  constexpr ExtraAttrPropertySet(uint64_t bitset) : bitset_(bitset) {}
  uint64_t bitset_;
};

const std::unordered_map<std::string, ExtraAttrPropertySet>
    extra_attr_properties = {
        // DEPRECATED attributes
        {"use_quantizer", ExtraAttrProperty::DEPRECATED},
        // SCHEDULE attributes
        {"use_cudnn", ExtraAttrProperty::SCHEDULE},
        {"use_mkldnn", ExtraAttrProperty::SCHEDULE},
        // ONEDNN dedicated attributes
        {"data_format", ExtraAttrProperty::ONEDNN},
        {"force_fp32_output", ExtraAttrProperty::ONEDNN},
        {"fuse_activation", ExtraAttrProperty::ONEDNN},
        {"fuse_activation_type", ExtraAttrProperty::ONEDNN},
        {"fuse_activation_alpha", ExtraAttrProperty::ONEDNN},
        {"fuse_activation_beta", ExtraAttrProperty::ONEDNN},
        {"fuse_activation_scale", ExtraAttrProperty::ONEDNN},
        {"fused_output_scale", ExtraAttrProperty::ONEDNN},
        {"fuse_alpha", ExtraAttrProperty::ONEDNN},
        {"fuse_beta", ExtraAttrProperty::ONEDNN},
        {"fuse_relu", ExtraAttrProperty::ONEDNN},
        {"fused_output_scale", ExtraAttrProperty::ONEDNN},
        {"fuse_residual_connection", ExtraAttrProperty::ONEDNN},
        {"fuse_with_relu", ExtraAttrProperty::ONEDNN},
        {"fused_reshape_Out", ExtraAttrProperty::ONEDNN},
        {"fused_transpose_Out", ExtraAttrProperty::ONEDNN},
        {"fused_reshape_X", ExtraAttrProperty::ONEDNN},
        {"fused_reshape_Y", ExtraAttrProperty::ONEDNN},
        {"fused_transpose_X", ExtraAttrProperty::ONEDNN},
        {"fused_transpose_Y", ExtraAttrProperty::ONEDNN},
        {"mkldnn_data_type", ExtraAttrProperty::ONEDNN},
        {"scale_x", ExtraAttrProperty::ONEDNN},
        {"scale_y", ExtraAttrProperty::ONEDNN},
        {"scale_out", ExtraAttrProperty::ONEDNN},
        {"Scale_in", ExtraAttrProperty::ONEDNN},
        {"Scale_in_eltwise", ExtraAttrProperty::ONEDNN},
        {"Scale_x", ExtraAttrProperty::ONEDNN},
        {"Scale_y", ExtraAttrProperty::ONEDNN},
        {"Scale_out", ExtraAttrProperty::ONEDNN},
        {"Scale_weights", ExtraAttrProperty::ONEDNN},
        {"x_data_format", ExtraAttrProperty::ONEDNN},
        {"y_data_format", ExtraAttrProperty::ONEDNN},
        {"fused_squeeze2_axes", ExtraAttrProperty::ONEDNN},
        {"fused_unsqueeze2_axes", ExtraAttrProperty::ONEDNN},
        {"fused_reshape2_shape", ExtraAttrProperty::ONEDNN},
        // ONEDNN pass dedicated attributes
        {"Activation_scale", ExtraAttrProperty::ONEDNN},
        {"Bias_scales", ExtraAttrProperty::ONEDNN},
        {"Output_shift_scale", ExtraAttrProperty::ONEDNN},
        {"Sum_scale", ExtraAttrProperty::ONEDNN},
        // GPUDNN dedicated attributes
        {"exhaustive_search", ExtraAttrProperty::GPUDNN},
        {"fuse_relu_before_depthwise_conv", ExtraAttrProperty::GPUDNN},
        {"use_addto", ExtraAttrProperty::GPUDNN},
        {"workspace_size_MB", ExtraAttrProperty::GPUDNN},
        // Mixed-use attributes
        {"is_test",
         ExtraAttrPropertySet(ExtraAttrProperty::ONEDNN) |
             ExtraAttrPropertySet(ExtraAttrProperty::GPUDNN)},
};

inline ExtraAttrPropertySet GetExtraAttrProperties(
    const std::string& attr_name) {
  auto iter = extra_attr_properties.find(attr_name);
  if (iter != extra_attr_properties.end()) {
    return iter->second;
  }
  return ExtraAttrPropertySet();
}

template <typename T>
struct ExtraAttrChecker {
  ExtraAttrChecker(const std::string& attr_name, T default_value)
      : attr_name(attr_name), default_val(default_value) {}

  void operator()(framework::AttributeMap* attr_map,
                  bool only_check_exist_value) {
    auto it = attr_map->find(attr_name);
    if (it == attr_map->end()) {
      if (!only_check_exist_value) {
        attr_map->emplace(attr_name, default_val);
      }
      return;
    }
    framework::ExtractAttribute<T> extract_attr(attr_name);
    extract_attr(it->second);
  }

  const std::string& attr_name;
  T default_val;
};

class ExtraInfoUtils {
 public:
  static ExtraInfoUtils& Instance() {
    static ExtraInfoUtils extra_info_utils;
    return extra_info_utils;
  }

  const std::unordered_map<std::string, paddle::framework::AttributeMap>&
  GetAllExtraAttrsMap() const {
    return g_extra_attrs_map_;
  }

  const paddle::framework::AttributeMap& GetExtraAttrsMap(
      const std::string& op_type) const {
    auto iter = g_extra_attrs_map_.find(op_type);
    if (iter != g_extra_attrs_map_.end()) {
      return iter->second;
    }
    return empty_extra_attrs_map_;
  }

  const std::vector<std::function<void(framework::AttributeMap*, bool)>>&
  GetExtraAttrsChecker(const std::string& op_type) const {
    auto iter = g_extra_attrs_checker_.find(op_type);
    if (iter != g_extra_attrs_checker_.end()) {
      return iter->second;
    }
    return empty_extra_attrs_checker_;
  }

  const std::vector<std::string>& GetExtraInputNamesMap(
      const std::string& op_type) const {
    auto iter = g_extra_input_names_map_.find(op_type);
    if (iter != g_extra_input_names_map_.end()) {
      return iter->second;
    }
    return empty_extra_input_names_;
  }

 private:
  ExtraInfoUtils();

  std::unordered_map<std::string, paddle::framework::AttributeMap>
      g_extra_attrs_map_;
  paddle::framework::AttributeMap empty_extra_attrs_map_{};
  std::unordered_map<
      std::string,
      std::vector<std::function<void(framework::AttributeMap*, bool)>>>
      g_extra_attrs_checker_;
  std::vector<std::function<void(framework::AttributeMap*, bool)>>
      empty_extra_attrs_checker_{};

  // TODO(chenweihang): move these extra inputs into op_compat.yaml
  std::unordered_map<std::string, std::vector<std::string>>
      g_extra_input_names_map_ = {{"conv2d", {"Bias", "ResidualData"}},
                                  {"conv2d_transpose", {"Bias"}},
                                  {"conv2d_grad", {"Bias"}},
                                  {"matmul_v2", {"ResidualData"}}};
  std::vector<std::string> empty_extra_input_names_;
};

}  // namespace operators
}  // namespace paddle
