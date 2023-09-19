// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/cinn/frontend/paddle/cpp/op_desc.h"
#include "paddle/cinn/frontend/var_type_utils.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace frontend {
namespace utils {

template <typename T>
inline T GetAttrOrDefault(const paddle::cpp::OpDesc& op_desc,
                          const std::string& name,
                          const T& default_value = T{}) {
  if (op_desc.HasAttr(name)) {
    return op_desc.GetAttr<T>(name);
  }
  return default_value;
}

#define EXPAND_SINGLE_NUM_TO_VECTOR(DATA_TYPE, ATTR_TYPE)                    \
  template <>                                                                \
  inline std::vector<DATA_TYPE> GetAttrOrDefault(                            \
      const paddle::cpp::OpDesc& op_desc,                                    \
      const std::string& name,                                               \
      const std::vector<DATA_TYPE>& default_value) {                         \
    if (op_desc.HasAttr(name)) {                                             \
      auto attr_type = op_desc.GetAttrType(name);                            \
      using AttrType = paddle::cpp::OpDescAPI::AttrType;                     \
      switch (attr_type) {                                                   \
        case AttrType::ATTR_TYPE##S:                                         \
          return op_desc.GetAttr<std::vector<DATA_TYPE>>(name);              \
        case AttrType::ATTR_TYPE:                                            \
          return std::vector<DATA_TYPE>{op_desc.GetAttr<DATA_TYPE>(name)};   \
        default:                                                             \
          if (attr_type == AttrType::BOOLEANS) {                             \
            LOG(WARNING) << "Op \"" << op_desc.Type() << "\"'s attribute \"" \
                         << name << "\" should be " << #ATTR_TYPE            \
                         << "S, but here is BOOLEANS, considering the type " \
                            "of python empty list in cpp are BOOLEANS,"      \
                         << " here we will return a empty vector.";          \
            return {};                                                       \
          } else {                                                           \
            LOG(FATAL) << "Op \"" << op_desc.Type() << "\"'s attribute \""   \
                       << name << "\" should be " << #ATTR_TYPE              \
                       << "S. But here " << static_cast<int>(attr_type)      \
                       << " Please Check!";                                  \
          }                                                                  \
      }                                                                      \
    }                                                                        \
    return default_value;                                                    \
  }

EXPAND_SINGLE_NUM_TO_VECTOR(int, INT)
EXPAND_SINGLE_NUM_TO_VECTOR(float, FLOAT)
EXPAND_SINGLE_NUM_TO_VECTOR(std::string, STRING)
EXPAND_SINGLE_NUM_TO_VECTOR(bool, BOOLEAN)

#undef EXPAND_SINGLE_NUM_TO_VECTOR

template <>
inline bool GetAttrOrDefault(const paddle::cpp::OpDesc& op_desc,
                             const std::string& name,
                             const bool& default_value) {
  if (op_desc.HasAttr(name)) {
    auto attr_type = op_desc.GetAttrType(name);
    using AttrType = paddle::cpp::OpDescAPI::AttrType;
    switch (attr_type) {
      case AttrType::BOOLEAN:
        return op_desc.GetAttr<bool>(name);
      case AttrType::INT:
        return static_cast<bool>(op_desc.GetAttr<int>(name));
      case AttrType::LONG:
        return static_cast<bool>(op_desc.GetAttr<int64_t>(name));
      default:
        LOG(FATAL) << "Op " << op_desc.Type() << "'s attribute " << name
                   << " should be BOOLEAN. Please Check!";
    }
  }
  return default_value;
}

template <>
inline int64_t GetAttrOrDefault(const paddle::cpp::OpDesc& op_desc,
                                const std::string& name,
                                const int64_t& default_value) {
  if (op_desc.HasAttr(name)) {
    auto attr_type = op_desc.GetAttrType(name);
    using AttrType = paddle::cpp::OpDescAPI::AttrType;
    switch (attr_type) {
      case AttrType::LONG:
        return op_desc.GetAttr<int64_t>(name);
      case AttrType::INT:
        return static_cast<int64_t>(op_desc.GetAttr<int>(name));
      default:
        LOG(FATAL) << "Op " << op_desc.Type() << "'s attribute " << name
                   << " should be LONG. Please Check!";
    }
  }
  return default_value;
}

template <>
inline std::vector<int64_t> GetAttrOrDefault(
    const paddle::cpp::OpDesc& op_desc,
    const std::string& name,
    const std::vector<int64_t>& default_value) {
  if (op_desc.HasAttr(name)) {
    auto attr_type = op_desc.GetAttrType(name);
    using AttrType = paddle::cpp::OpDescAPI::AttrType;
    switch (attr_type) {
      case AttrType::LONGS:
        return op_desc.GetAttr<std::vector<int64_t>>(name);
      case AttrType::LONG:
        return std::vector<int64_t>{GetAttrOrDefault<int64_t>(op_desc, name)};
      case AttrType::INTS: {
        const auto& ints_val =
            GetAttrOrDefault<std::vector<int>>(op_desc, name);
        return std::vector<int64_t>{ints_val.begin(), ints_val.end()};
      }
      case AttrType::INT:
        return std::vector<int64_t>{GetAttrOrDefault<int>(op_desc, name)};
      case AttrType::BOOLEANS: {
        LOG(WARNING) << "Op \"" << op_desc.Type() << "\"'s attribute \"" << name
                     << "\" should be LONGS, "
                     << "but here is BOOLEANS, considering the type of python "
                        "empty list in cpp are BOOLEANS, "
                     << "here we will return a empty vector.";
        return {};
      }
      default:
        LOG(FATAL) << "Op " << op_desc.Type() << "'s attribute " << name
                   << " should be LONGS. Please Check!";
    }
  }
  return default_value;
}

template <typename T>
inline cinn::utils::ShapeType ToShapeType(const std::vector<T>& shape) {
  return cinn::utils::ShapeType(shape.begin(), shape.end());
}

template <typename T>
inline cinn::utils::DimType ToDimType(const T& val) {
  return static_cast<cinn::utils::DimType>(val);
}

inline std::string GetPaddleDtype(const paddle::cpp::OpDesc& op_desc,
                                  const std::string& dtype_attr_name,
                                  paddle::cpp::VarDescAPI::Type default_dtype) {
  auto dtype_id = GetAttrOrDefault<int>(
      op_desc, dtype_attr_name, static_cast<int>(default_dtype));
  if (dtype_id < 0) {
    return "";
  }
  auto dtype_pd = static_cast<paddle::cpp::VarDescAPI::Type>(dtype_id);
  auto dtype_cinn = CppVarType2CommonType(dtype_pd);
  if (dtype_cinn.is_unk()) {
    return "";
  }

  return common::Type2Str(dtype_cinn);
}

}  // namespace utils
}  // namespace frontend
}  // namespace cinn
