/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dtu/hlir/builder/hlir_builder.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/platform/device/gcu/layout/gcu_layout_interface.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
#include "paddle/fluid/platform/device/gcu/utils/utils.h"

namespace paddle {
namespace platform {
namespace gcu {
typedef enum { CHANNELFIRST = 0, CHANNELLAST, INSENSITIVE } LayoutType;

using EquivalenceTransformFunc = std::function<GcuOpPtr(
    GcuBuilderPtr builder,
    const NodePtr &node,
    std::map<std::string, std::vector<GcuOpPtr>> map_inputs,
    std::string running_mode)>;

struct OpInfo {
  OpInfo() = default;
  OpInfo(const OpInfo &op_info) = default;
  OpInfo &operator=(const OpInfo &op_info) = default;
  ~OpInfo() = default;

  std::map<std::string, Layout> ins_layouts;
  std::map<std::string, Layout> outs_layouts;
  std::map<std::string, std::unordered_set<std::string>> ins_datatype;
  std::map<std::string, std::unordered_set<std::string>> outs_datatype;
};

#define UNIQUE_NAME(A, B) _gcu_equivalence_trans_register_##A##_##B##_
#define IMPLEMT_EQUIVALENCE_TRANS_FUNC(                        \
    builder, node, map_inputs, running_mode, func)             \
  static GcuOpPtr _##func##_INSENSITIVE(                       \
      GcuBuilderPtr builder,                                   \
      const NodePtr &node,                                     \
      std::map<std::string, std::vector<GcuOpPtr>> map_inputs, \
      std::string running_mode = "serial")

#define IMPLEMT_EQUIVALENCE_TRANS_CFIRST_FUNC(                 \
    builder, node, map_inputs, running_mode, func)             \
  static GcuOpPtr _##func##_CHANNELFIRST(                      \
      GcuBuilderPtr builder,                                   \
      const NodePtr &node,                                     \
      std::map<std::string, std::vector<GcuOpPtr>> map_inputs, \
      std::string running_mode = "serial")

#define IMPLEMT_EQUIVALENCE_TRANS_CLAST_FUNC(                  \
    builder, node, map_inputs, running_mode, func)             \
  static GcuOpPtr _##func##_CHANNELLAST(                       \
      GcuBuilderPtr builder,                                   \
      const NodePtr &node,                                     \
      std::map<std::string, std::vector<GcuOpPtr>> map_inputs, \
      std::string format_flag = "NCHW")

#define EQUIVALENCE_TRANS_FUNC_REG(op_type, adapter_type, func)         \
  static EquivalenceTransWrapperObj UNIQUE_NAME(op_type, adapter_type)( \
      op_type, adapter_type, _##func##_##adapter_type);

#define EQUIVALENCE_TRANS_FUNC_REG_WITH_OP_INFO(                        \
    op_type, adapter_type, func, ...)                                   \
  static EquivalenceTransWrapperObj UNIQUE_NAME(op_type, adapter_type)( \
      op_type, adapter_type, _##func##_##adapter_type, __VA_ARGS__);

class EquivalenceTransformer {
 public:
  static EquivalenceTransformer &GetInstance() {
    static EquivalenceTransformer trans;
    return trans;
  }

  void Registry(const std::string &op_type,
                LayoutType type,
                EquivalenceTransformFunc adapter_func) {
    std::lock_guard<std::mutex> lock(mu_);
    switch (type) {
      case CHANNELFIRST:
        channel_first_funcs_[op_type] = adapter_func;
        break;
      case INSENSITIVE:
        insensitive_funcs_[op_type] = adapter_func;
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupport registered layout type when register gcu op!"));
    }
  }

  void Registry(const std::string &op_type,
                LayoutType type,
                EquivalenceTransformFunc adapter_func,
                const OpInfo &op_info) {
    std::lock_guard<std::mutex> lock(mu_);
    switch (type) {
      case CHANNELLAST:
        channel_last_funcs_[op_type] = adapter_func;
        op_infos_[op_type + std::to_string(static_cast<int32_t>(type))] =
            op_info;
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupport registered layout type when register gcu op!"));
    }
  }

  EquivalenceTransformFunc Get(const std::string &op_type,
                               const LayoutType &type = CHANNELFIRST) {
    std::lock_guard<std::mutex> lock(mu_);
    if (type == CHANNELFIRST) {
      auto iter = channel_first_funcs_.find(op_type);
      if (iter == channel_first_funcs_.end()) {
        return nullptr;
      }
      return iter->second;
    } else if (type == CHANNELLAST) {
      auto iter = channel_last_funcs_.find(op_type);
      if (iter == channel_last_funcs_.end()) {
        return nullptr;
      }
      return iter->second;
    }
    auto iter = insensitive_funcs_.find(op_type);
    if (iter == insensitive_funcs_.end()) {
      return nullptr;
    }
    return iter->second;
  }

  OpInfo GetOpInfo(const std::string &op_type, LayoutType type) {
    std::lock_guard<std::mutex> lock(mu_);
    return op_infos_[op_type + std::to_string(static_cast<int32_t>(type))];
  }

 public:
  std::mutex mu_;
  std::unordered_set<std::string> registerd_ops_;
  std::unordered_map<std::string, EquivalenceTransformFunc>
      channel_first_funcs_;
  std::unordered_map<std::string, EquivalenceTransformFunc> channel_last_funcs_;
  std::unordered_map<std::string, OpInfo> op_infos_;
  std::unordered_map<std::string, EquivalenceTransformFunc> insensitive_funcs_;
};

struct EquivalenceTransWrapperObj {
  EquivalenceTransWrapperObj(const std::string &op_type,
                             LayoutType type,
                             EquivalenceTransformFunc func) {
    EquivalenceTransformer::GetInstance().Registry(op_type, type, func);
  }
  EquivalenceTransWrapperObj(const std::string &op_type,
                             LayoutType type,
                             EquivalenceTransformFunc func,
                             const OpInfo &op_info) {
    EquivalenceTransformer::GetInstance().Registry(
        op_type, type, func, op_info);
  }
  ~EquivalenceTransWrapperObj() = default;
};
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
