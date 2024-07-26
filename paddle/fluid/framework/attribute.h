/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <stdint.h>

#include <functional>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/any.h"
#include "paddle/utils/variant.h"

namespace paddle {
namespace framework {

TEST_API paddle::any GetAttrValue(const Attribute& attr);

TEST_API Attribute GetAttrValue(const proto::OpDesc::Attr& attr_desc);

Attribute GetAttrValue(const proto::VarDesc::Attr& attr_desc);

template <typename T>
struct ExtractAttribute {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  T* operator()(Attribute& attr) const {
    T* attr_value = nullptr;
    try {
      attr_value = &paddle::get<T>(attr);
    } catch (paddle::bad_variant_access const& bad_get) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cannot get attribute (%s) by type %s, its type is %s.",
          attr_name_,
          paddle::platform::demangle(typeid(T).name()),
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

// special handle bool
// FIXME(yuyang18): Currently we cast bool into int in python binding. It is
// hard to change the logic there. In another way, we should correct handle
// if the user set `some_flag=1`.
//
// FIX ME anytime if there is a better solution.
template <>
struct ExtractAttribute<bool> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  bool* operator()(Attribute& attr) const {
    if (attr.type() == typeid(int)) {  // NOLINT
      int val = PADDLE_GET_CONST(int, attr);
      attr = static_cast<bool>(val);
    } else if (attr.type() == typeid(float)) {  // NOLINT
      float val = PADDLE_GET_CONST(float, attr);
      attr = static_cast<bool>(val);
    }
    bool* attr_value = nullptr;
    try {
      attr_value = &paddle::get<bool>(attr);
    } catch (paddle::bad_variant_access const& bad_get) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cannot get attribute (%s) by type bool, its type is %s.",
          attr_name_,
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<int64_t> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  int64_t* operator()(Attribute& attr) const {
    if (attr.type() == typeid(int)) {  // NOLINT
      int val = PADDLE_GET_CONST(int, attr);
      attr = static_cast<int64_t>(val);
    } else if (attr.type() == typeid(float)) {  // NOLINT
      int val = PADDLE_GET_CONST(float, attr);
      attr = static_cast<int64_t>(val);
    }
    int64_t* attr_value = nullptr;
    try {
      attr_value = &paddle::get<int64_t>(attr);
    } catch (paddle::bad_variant_access const& bad_get) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cannot get attribute (%s) by type int64_t, its type is %s.",
          attr_name_,
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<std::vector<int64_t>> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  std::vector<int64_t>* operator()(Attribute& attr) const {
    if (attr.type() == typeid(std::vector<int>)) {  // NOLINT
      std::vector<int> val = PADDLE_GET_CONST(std::vector<int>, attr);
      std::vector<int64_t> vec(val.begin(), val.end());
      attr = vec;
    } else if (attr.type() == typeid(std::vector<float>)) {  // NOLINT
      std::vector<float> val = PADDLE_GET_CONST(std::vector<float>, attr);
      std::vector<int64_t> vec(val.begin(), val.end());
      attr = vec;
    }
    std::vector<int64_t>* attr_value = nullptr;
    try {
      attr_value = &paddle::get<std::vector<int64_t>>(attr);
    } catch (paddle::bad_variant_access const& bad_get) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cannot get attribute (%s) by type std::vector<int64_t>, its type is "
          "%s.",
          attr_name_,
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<float> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  float* operator()(Attribute& attr) const {
    if (attr.type() == typeid(int)) {  // NOLINT
      int val = PADDLE_GET_CONST(int, attr);
      attr = static_cast<float>(val);
    } else if (attr.type() == typeid(int64_t)) {  // NOLINT
      int64_t val = PADDLE_GET_CONST(int64_t, attr);
      attr = static_cast<float>(val);
    }
    float* attr_value = nullptr;
    try {
      attr_value = &paddle::get<float>(attr);
    } catch (paddle::bad_variant_access const& bad_get) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cannot get attribute (%s) by type float, its type is %s.",
          attr_name_,
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<double> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  double* operator()(Attribute& attr) const {
    if (attr.type() == typeid(int)) {  // NOLINT
      int val = PADDLE_GET_CONST(int, attr);
      attr = static_cast<double>(val);
    } else if (attr.type() == typeid(int64_t)) {  // NOLINT
      int64_t val = PADDLE_GET_CONST(int64_t, attr);
      attr = static_cast<double>(val);
    } else if (attr.type() == typeid(float)) {  // NOLINT
      int64_t val = PADDLE_GET_CONST(float, attr);
      attr = static_cast<double>(val);
    }
    double* attr_value = nullptr;
    try {
      attr_value = &paddle::get<double>(attr);
    } catch (paddle::bad_variant_access const& bad_get) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cannot get attribute (%s) by type double, its type is %s.",
          attr_name_,
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<std::vector<double>> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  std::vector<double>* operator()(Attribute& attr) const {
    if (attr.type() == typeid(std::vector<int>)) {  // NOLINT
      std::vector<int> val = PADDLE_GET_CONST(std::vector<int>, attr);
      std::vector<double> vec(val.begin(), val.end());
      attr = vec;
    } else if (attr.type() == typeid(std::vector<float>)) {  // NOLINT
      std::vector<float> val = PADDLE_GET_CONST(std::vector<float>, attr);
      std::vector<double> vec(val.begin(), val.end());
      attr = vec;
    }
    std::vector<double>* attr_value = nullptr;
    try {
      attr_value = &paddle::get<std::vector<double>>(attr);
    } catch (paddle::bad_variant_access const& bad_get) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cannot get attribute (%s) by type std::vector<double>, its type is "
          "%s.",
          attr_name_,
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<paddle::experimental::Scalar> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  paddle::experimental::Scalar* operator()(Attribute& attr) const {
    paddle::experimental::Scalar* attr_value = nullptr;
    try {
      attr_value = &paddle::get<paddle::experimental::Scalar>(attr);
    } catch (paddle::bad_variant_access const& bad_get) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Cannot get attribute (%s) by type Scalar, its type is %s, index is "
          "%d",
          attr_name_,
          paddle::platform::demangle(attr.type().name()),
          attr.index()));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <typename T>
inline proto::AttrType AttrTypeID() {
  Attribute tmp = T();
  return static_cast<proto::AttrType>(tmp.index() - 1);
}

inline proto::AttrType AttrTypeID(const Attribute& attr) {
  return static_cast<proto::AttrType>(attr.index() - 1);
}

inline bool IsAttrVar(const Attribute& attr) {
  return AttrTypeID(attr) == proto::AttrType::VAR;
}

inline bool IsAttrVars(const Attribute& attr) {
  return AttrTypeID(attr) == proto::AttrType::VARS;
}

inline bool HasAttrVar(const Attribute& attr) {
  return IsAttrVar(attr) || IsAttrVars(attr);
}

inline AttributeMap FilterAttrVar(const AttributeMap& attrs) {
  AttributeMap attrs_var;
  for (auto& attr : attrs) {
    if (HasAttrVar(attr.second)) {
      attrs_var.emplace(attr);
    }
  }
  return attrs_var;
}

class AttrReader {
 public:
  explicit AttrReader(const AttributeMap& attrs)
      : attrs_(attrs), default_attrs_(nullptr) {}

  AttrReader(const AttributeMap& attrs, const AttributeMap& default_attrs)
      : attrs_(attrs), default_attrs_(&default_attrs) {}

  template <typename T>
  inline const T& Get(const std::string& name) const {
    auto it = attrs_.find(name);
    bool found = it != attrs_.end();
    if (!found) {
      if (default_attrs_ != nullptr) {
        it = default_attrs_->find(name);
        found = it != default_attrs_->end();
      }
    }
    PADDLE_ENFORCE_EQ(found,
                      true,
                      phi::errors::NotFound(
                          "Attribute (%s) should be in AttributeMap.", name));

    Attribute& attr = const_cast<Attribute&>(it->second);
    ExtractAttribute<T> extract_attr(name);
    T* attr_value = extract_attr(attr);
    return *attr_value;
  }

  const Attribute* GetAttr(const std::string& name) const {
    auto it = attrs_.find(name);
    bool found = it != attrs_.end();
    if (!found) {
      if (default_attrs_ != nullptr) {
        it = default_attrs_->find(name);
        found = it != default_attrs_->end();
      }
    }
    if (found) {
      return &it->second;
    }
    return nullptr;
  }

 private:
  const AttributeMap& attrs_;
  const AttributeMap* default_attrs_;
};

paddle::experimental::Scalar MakeScalarFromProto(const proto::Scalar& v);
TEST_API proto::Scalar MakeScalarProto(const paddle::experimental::Scalar& v);
TEST_API paddle::experimental::Scalar MakeScalarFromAttribute(
    const Attribute& v);
TEST_API std::vector<paddle::experimental::Scalar> MakeScalarsFromAttribute(
    const Attribute& v);
void CanonicalizeScalarAttrs(const proto::OpProto& op_proto,
                             AttributeMap* attrs);
}  // namespace framework
}  // namespace paddle
