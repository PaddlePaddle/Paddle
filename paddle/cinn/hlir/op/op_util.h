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
#include <absl/container/flat_hash_map.h>

#include <string>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace hlir {

template <typename T>
T GetAttr(const cinn::utils::AttributeMap &attr_map,
          const std::string &attr_name) {
  PADDLE_ENFORCE_EQ(attr_map.count(attr_name),
                    true,
                    ::common::errors::InvalidArgument(
                        "Sorry, cannot found attribute %s", attr_name));
  const auto &attr = attr_map.at(attr_name);
  PADDLE_ENFORCE_EQ(
      absl::holds_alternative<T>(attr),
      true,
      ::common::errors::InvalidArgument(
          "The type of attribute %s isn't %s", attr_name, typeid(T).name()));
  return absl::get<T>(attr_map.at(attr_name));
}

template <class T>
T SafeGetAttr(const cinn::utils::AttributeMap &attrs,
              const std::string &key,
              const T &&value) {
  if (attrs.find(key) != attrs.end()) {
    return GetAttr<T>(attrs, key);
  }
  return value;
}

template <typename T = int>
std::vector<Expr> ToCinnExprs(const std::vector<T> &args) {
  std::vector<Expr> exprs;
  std::transform(
      args.begin(), args.end(), std::back_inserter(exprs), [](const T &arg) {
        return Expr(arg);
      });
  return exprs;
}

std::vector<Expr> ToCinnExprs(const std::vector<ir::Dim> &args);

template <typename T>
std::vector<T> ToPodVector(const std::vector<Expr> &args) {
  if (args.empty()) {
    return {};
  }

  const auto &type = args.front().type();
  PADDLE_ENFORCE_EQ(
      type,
      cinn::common::type_of<T>(),
      ::common::errors::InvalidArgument("Cannot get %s value from %s vector!",
                                        cinn::common::type_of<T>(),
                                        type));

  std::vector<T> shape_v;
  if (type.is_bool()) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_bool()));
    }
  } else if (type.is_int(8)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_int8()));
    }
  } else if (type.is_int(16)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_int16()));
    }
  } else if (type.is_int(32)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_int32()));
    }
  } else if (type.is_int(64)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_int64()));
    }
  } else if (type.is_uint(8)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_uint8()));
    }
  } else if (type.is_uint(16)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_uint16()));
    }
  } else if (type.is_uint(32)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_uint32()));
    }
  } else if (type.is_uint(64)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_uint64()));
    }
  } else if (type.is_bfloat16()) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_bfloat16()));
    }
  } else if (type.is_float16()) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_float16()));
    }
  } else if (type.is_float(32)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_float()));
    }
  } else if (type.is_float(64)) {
    for (auto &e : args) {
      shape_v.push_back(static_cast<T>(e.as_double()));
    }
  } else {
    std::stringstream ss;
    ss << "Not support " << type;
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));
  }
  return shape_v;
}

using CINNSchedule = lang::PackedFunc;

CINNSchedule GetElementwiseScheduleFunc(
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target,
    bool vectorizable = true);

CINNSchedule GetInjectiveScheduleFunc(
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target,
    bool vectorizable = true);

std::string GetExternFuncName(const cinn::common::Target &target,
                              const cinn::common::Type &type,
                              const std::string &func_name,
                              const bool need_cinn = true,
                              const bool need_target = true,
                              const bool need_type = true);

}  // namespace hlir
}  // namespace cinn
