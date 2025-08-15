// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <ATen/tensor.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

namespace torch {

enum class TypeTag { None = 0, Bool, Int, Double, String, Tensor };

class IValue {
 public:
  IValue() : tag_(TypeTag::None), value_() {}

  IValue(bool val) : tag_(TypeTag::Bool), value_(val) {}      // NOLINT
  IValue(int val) : tag_(TypeTag::Int), value_(val) {}        // NOLINT
  IValue(double val) : tag_(TypeTag::Double), value_(val) {}  // NOLINT
  IValue(const std::string& val)                              // NOLINT
      : tag_(TypeTag::String), value_(val) {}
  IValue(std::string&& val)  // NOLINT
      : tag_(TypeTag::String), value_(std::move(val)) {}
  IValue(const char* val)  // NOLINT
      : tag_(TypeTag::String), value_(std::string(val)) {}
  IValue(at::Tensor val) : tag_(TypeTag::Tensor), value_(val) {}  // NOLINT

  IValue(const IValue& other) = default;
  IValue(IValue&& other) = default;
  IValue& operator=(const IValue& other) = default;
  IValue& operator=(IValue&& other) = default;

  bool is_none() const { return tag_ == TypeTag::None; }
  bool is_bool() const { return tag_ == TypeTag::Bool; }
  bool is_int() const { return tag_ == TypeTag::Int; }
  bool is_double() const { return tag_ == TypeTag::Double; }
  bool is_string() const { return tag_ == TypeTag::String; }
  bool is_tensor() const { return tag_ == TypeTag::Tensor; }

  bool to_bool() const {
    if (!is_bool()) throw std::runtime_error("Not a bool");
    return std::get<bool>(value_);
  }

  int to_int() const {
    if (!is_int()) throw std::runtime_error("Not an int");
    return std::get<int>(value_);
  }

  double to_double() const {
    if (!is_double()) throw std::runtime_error("Not a double");
    return std::get<double>(value_);
  }

  const std::string& to_string() const {
    if (!is_string()) throw std::runtime_error("Not a string");
    return std::get<std::string>(value_);
  }

  at::Tensor to_tensor() const {
    if (!is_tensor()) throw std::runtime_error("Not a tensor");
    return std::get<at::Tensor>(value_);
  }

  bool try_to_bool(bool& out) const {  // NOLINT
    if (is_bool()) {
      out = std::get<bool>(value_);
      return true;
    } else if (is_int()) {
      out = (std::get<int>(value_) != 0);
      return true;
    } else if (is_double()) {
      out = (std::get<double>(value_) != 0.0);
      return true;
    }
    return false;
  }

  bool try_to_int(int& out) const {  // NOLINT
    if (is_int()) {
      out = std::get<int>(value_);
      return true;
    } else if (is_double()) {
      double val = std::get<double>(value_);
      if (val != static_cast<int>(val)) {
        std::cout << "Warning: Converting double(" << val
                  << ") to int (precision loss)" << std::endl;
      }
      out = static_cast<int>(val);
      return true;
    }
    return false;
  }

  bool try_to_double(double& out) const {  // NOLINT
    if (is_double()) {
      out = std::get<double>(value_);
      return true;
    } else if (is_int()) {
      out = static_cast<double>(std::get<int>(value_));
      return true;
    }
    return false;
  }

  bool try_to_string(std::string& out) const {  // NOLINT
    if (is_string()) {
      out = std::get<std::string>(value_);
      return true;
    }
    return false;
  }

  bool try_to_tensor(at::Tensor& out) const {  // NOLINT
    if (is_tensor()) {
      out = std::get<at::Tensor>(value_);
      return true;
    }
    return false;
  }

  template <typename T>
  bool try_convert_to(T& out) const {  // NOLINT
    // Remove reference and cv-qualifiers from T
    using BaseType = std::remove_cv_t<std::remove_reference_t<T>>;

    if constexpr (std::is_same_v<BaseType, bool>) {
      return try_to_bool(const_cast<bool&>(reinterpret_cast<const bool&>(out)));
    } else if constexpr (std::is_same_v<BaseType, int>) {
      return try_to_int(const_cast<int&>(reinterpret_cast<const int&>(out)));
    } else if constexpr (std::is_same_v<BaseType, double>) {
      return try_to_double(
          const_cast<double&>(reinterpret_cast<const double&>(out)));
    } else if constexpr (std::is_same_v<BaseType, std::string>) {
      return try_to_string(
          const_cast<std::string&>(reinterpret_cast<const std::string&>(out)));
    } else if constexpr (std::is_same_v<BaseType, at::Tensor>) {
      return try_to_tensor(
          const_cast<at::Tensor&>(reinterpret_cast<const at::Tensor&>(out)));
    } else {
      return false;
    }
  }

  std::string type_string() const {
    switch (tag_) {
      case TypeTag::None:
        return "None";
      case TypeTag::Bool:
        return "Bool";
      case TypeTag::Int:
        return "Int";
      case TypeTag::Double:
        return "Double";
      case TypeTag::String:
        return "String";
      case TypeTag::Tensor:
        return "Tensor";
      default:
        return "Unknown";
    }
  }

  std::string to_repr() const {
    switch (tag_) {
      case TypeTag::None:
        return "None";
      case TypeTag::Bool:
        return std::get<bool>(value_) ? "true" : "false";
      case TypeTag::Int:
        return std::to_string(std::get<int>(value_));
      case TypeTag::Double:
        return std::to_string(std::get<double>(value_));
      case TypeTag::String:
        return "\"" + std::get<std::string>(value_) + "\"";
      default:
        return "Unknown";
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const IValue& val) {
    return os << val.to_repr();
  }

 private:
  TypeTag tag_;
  std::variant<std::monostate, bool, int, double, std::string, at::Tensor>
      value_;
};

}  // namespace torch
