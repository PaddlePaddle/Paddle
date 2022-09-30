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

#include "paddle/fluid/framework/attribute.h"

namespace paddle {
namespace operators {

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
};

}  // namespace operators
}  // namespace paddle
