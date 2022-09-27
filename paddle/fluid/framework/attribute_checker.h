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
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {
// check whether a value(attribute) fit a certain limit
template <typename T>
class GreaterThanChecker {
 public:
  explicit GreaterThanChecker(T lower_bound) : lower_bound_(lower_bound) {}
  void operator()(const T& value) const {
    PADDLE_ENFORCE_GT(
        value,
        lower_bound_,
        platform::errors::OutOfRange("Check for attribute value greater than "
                                     "a certain value failed."));
  }

 private:
  T lower_bound_;
};

template <typename T>
class EqualGreaterThanChecker {
 public:
  explicit EqualGreaterThanChecker(T lower_bound) : lower_bound_(lower_bound) {}
  void operator()(const T& value) const {
    PADDLE_ENFORCE_GE(
        value,
        lower_bound_,
        platform::errors::OutOfRange("Check for attribute valur equal or "
                                     "greater than a certain value failed."));
  }

 private:
  T lower_bound_;
};

template <typename T>
class TypedAttrVarInfoChecker {
 public:
  TypedAttrVarInfoChecker() = default;

  void operator()(const Attribute& attr) const {
    if (IsAttrVar(attr)) {
      auto* var_desc = PADDLE_GET_CONST(VarDesc*, attr);
      check(var_desc);
    } else if (IsAttrVars(attr)) {
      auto var_descs = PADDLE_GET_CONST(std::vector<VarDesc*>, attr);
      check(var_descs);
    }
  }

  void check(const VarDesc* var_desc) const {
    PADDLE_ENFORCE_NOT_NULL(
        var_desc,
        platform::errors::InvalidArgument(
            "Required Attribute with Variable type shall not be nullptr."));
    auto shape = var_desc->GetShape();
    PADDLE_ENFORCE_EQ(shape.size(),
                      1U,
                      platform::errors::InvalidArgument(
                          "Required shape rank of Attribute(%s) == 1, "
                          "but received rank == %s",
                          var_desc->Name(),
                          shape.size()));

    auto& expected_type = typeid(T);
    auto dtype = var_desc->GetDataType();
    // attribute is a IntArray
    if (expected_type == typeid(std::vector<int64_t>) ||
        expected_type == typeid(std::vector<int>)) {
      bool is_int = (dtype == proto::VarType::Type::VarType_Type_INT32 ||
                     dtype == proto::VarType::Type::VarType_Type_INT64);
      PADDLE_ENFORCE_EQ(is_int,
                        true,
                        platform::errors::InvalidArgument(
                            "Required dtype of Attribute(%s) shall be "
                            "int32|int64, but recevied %s.",
                            var_desc->Name(),
                            dtype));
    }
  }

  void check(const std::vector<VarDesc*>& var_descs) const {
    for (auto& var_desc : var_descs) {
      PADDLE_ENFORCE_NOT_NULL(
          var_desc,
          platform::errors::InvalidArgument(
              "Required Attribute with Variable type shall not be nullptr."));
      auto shape = var_desc->GetShape();
      PADDLE_ENFORCE_EQ(shape.size(),
                        1U,
                        platform::errors::InvalidArgument(
                            "Required shape rank of Attribute(%s) == 1, "
                            "but received rank == %s",
                            var_desc->Name(),
                            shape.size()));
      PADDLE_ENFORCE_EQ(shape[0] == 1U || shape[0] == -1,
                        true,
                        platform::errors::InvalidArgument(
                            "Required shape[0] of Attribute(%s) == 1 or -1, "
                            "but received shape[0] == %s",
                            var_desc->Name(),
                            shape[0]));
    }
  }
};

// we can provide users more common Checker, like 'LessThanChecker',
// 'BetweenChecker'...

template <typename T>
class DefaultValueSetter {
 public:
  explicit DefaultValueSetter(T default_value)
      : default_value_(default_value) {}
  const T& operator()() const { return default_value_; }

 private:
  T default_value_;
};

template <typename T>
class EnumInContainer {
 public:
  explicit EnumInContainer(const std::unordered_set<T>& c) : container_(c) {}
  void operator()(const T& val) const {
    PADDLE_ENFORCE_NE(
        container_.find(val),
        container_.end(),
        platform::errors::NotFound("Value %s is not in enum container %s.",
                                   val,
                                   ContainerDebugString()));
  }

 private:
  std::string ContainerDebugString() const {
    std::ostringstream sout;
    sout << "[";
    size_t cnt = 0;
    for (auto& v : container_) {
      sout << v;
      ++cnt;
      if (cnt != container_.size()) {
        sout << " ,";
      }
    }
    sout << "]";
    return sout.str();
  }

  std::unordered_set<T> container_;
};

// check whether a certain attribute fit its limits
// an attribute can have more than one limits
template <typename T>
class TypedAttrChecker {
  typedef std::function<const T&()> DefaultValueChecker;
  typedef std::function<void(const T&)> ValueChecker;

 public:
  explicit TypedAttrChecker(const std::string& attr_name,
                            proto::OpProto_Attr* attr)
      : attr_name_(attr_name), attr_(attr) {}

  TypedAttrChecker& AsExtra() {
    attr_->set_extra(true);
    return *this;
  }

  TypedAttrChecker& AsQuant() {
    attr_->set_quant(true);
    return *this;
  }

  TypedAttrChecker& SupportTensor() {
    attr_->set_support_tensor(true);
    return *this;
  }

  TypedAttrChecker& InEnum(const std::unordered_set<T>& range) {
    value_checkers_.push_back(EnumInContainer<T>(range));
    return *this;
  }

  TypedAttrChecker& GreaterThan(const T& lower_bound) {
    value_checkers_.push_back(GreaterThanChecker<T>(lower_bound));
    return *this;
  }

  TypedAttrChecker& EqualGreaterThan(const T& lower_bound) {
    value_checkers_.push_back(EqualGreaterThanChecker<T>(lower_bound));
    return *this;
  }

  // we can add more common limits, like LessThan(), Between()...

  TypedAttrChecker& SetDefault(const T& default_value) {
    PADDLE_ENFORCE_EQ(
        default_value_setter_.empty(),
        true,
        platform::errors::AlreadyExists("Attribute (%s) has a default value "
                                        "and cannot be set repeatedly.",
                                        attr_name_));
    default_value_setter_.push_back(DefaultValueSetter<T>(default_value));
    return *this;
  }

  // allow users provide their own checker
  TypedAttrChecker& AddCustomChecker(const ValueChecker& checker) {
    value_checkers_.push_back(checker);
    return *this;
  }

  void operator()(AttributeMap* attr_map,
                  bool get_default_value_only = false,
                  bool only_check_exist_value = false) const {
    if (get_default_value_only) {
      if (!default_value_setter_.empty()) {
        attr_map->emplace(attr_name_, default_value_setter_[0]());
      }
      return;
    }
    // If attribute is VarDesc(s), we should verify it's supported in OpMaker
    auto it = attr_map->find(attr_name_);
    if (it != attr_map->end() && HasAttrVar(it->second)) {
      PADDLE_ENFORCE_EQ(attr_->support_tensor(),
                        true,
                        platform::errors::InvalidArgument(
                            "Found Attribute('%s') with type(Variable), but it "
                            "doesn't support Tensor type.",
                            attr_name_));

      VLOG(1) << "Found Attribute " << attr_name_ << " with type(Variable).";
      var_info_checker_(it->second);
      return;
    }

    if (only_check_exist_value) {
      if (it != attr_map->end()) {
        ExtractAttribute<T> extract_attr(attr_name_);
        T* attr_value = extract_attr(it->second);
        for (const auto& checker : value_checkers_) {
          checker(*attr_value);
        }
      }
    } else {
      if (it == attr_map->end()) {
        // user do not set this attr
        PADDLE_ENFORCE_EQ(
            default_value_setter_.empty(),
            false,
            platform::errors::InvalidArgument(
                "Attribute (%s) is not set correctly.", attr_name_));
        // default_value_setter_ has no more than one element
        auto tmp = attr_map->emplace(attr_name_, default_value_setter_[0]());
        it = tmp.first;
      }
      ExtractAttribute<T> extract_attr(attr_name_);
      T* attr_value = extract_attr(it->second);
      for (const auto& checker : value_checkers_) {
        checker(*attr_value);
      }
    }
  }

 private:
  std::string attr_name_;
  proto::OpProto_Attr* attr_;
  TypedAttrVarInfoChecker<T> var_info_checker_;
  std::vector<ValueChecker> value_checkers_;
  std::vector<DefaultValueChecker> default_value_setter_;
};

// check whether op's all attributes fit their own limits
class OpAttrChecker {
  typedef std::function<void(AttributeMap*, bool, bool)> AttrChecker;

 public:
  template <typename T>
  TypedAttrChecker<T>& AddAttrChecker(const std::string& attr_name,
                                      proto::OpProto_Attr* attr) {
    attr_checkers_.push_back(TypedAttrChecker<T>(attr_name, attr));
    AttrChecker& checker = attr_checkers_.back();
    return *(checker.target<TypedAttrChecker<T>>());
  }

  void Check(AttributeMap* attr_map,
             bool explicit_only = false,
             bool only_check_exist_value = false) const {
    auto checker_num = attr_checkers_.size();
    if (explicit_only) checker_num = explicit_checker_num_;
    for (size_t i = 0; i < checker_num; ++i) {
      attr_checkers_[i](attr_map, false, only_check_exist_value);
    }
  }

  AttributeMap GetDefaultAttrsMap() const {
    AttributeMap default_values_map;
    for (const auto& checker : attr_checkers_) {
      checker(&default_values_map, true, false);
    }
    return default_values_map;
  }

  void RecordExplicitCheckerNum() {
    explicit_checker_num_ = attr_checkers_.size();
  }

  void InitDefaultAttributeMap(const AttributeMap* extra_attr_map) {
    for (const auto& checker : attr_checkers_) {
      checker(&default_attrs_, true, false);
    }
    if (extra_attr_map) {
      default_attrs_.insert(extra_attr_map->begin(), extra_attr_map->end());
    }
  }

  const AttributeMap& GetDefaultAttrMap() const { return default_attrs_; }

 private:
  std::vector<AttrChecker> attr_checkers_;

  AttributeMap default_attrs_;

  // in order to improve the efficiency of dynamic graph mode,
  // we divede the attribute into explicit type and implicit type.
  // for explicit attribute, we mean the attribute added in the customized
  // op makers, usually it's defined in the overloaded Make method.
  // for implicit attribute, we mean the attribute added outside of the Make
  // method like "op_role", "op_role_var", and they are useless in dynamic
  // graph
  // mode
  size_t explicit_checker_num_;
};

}  // namespace framework
}  // namespace paddle
