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
#include <ATen/core/TensorBody.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

namespace torch {

class CustomClassHolder {
 public:
  virtual ~CustomClassHolder() = default;
};

template <typename T>
class intrusive_ptr {
 public:
  using element_type = T;
  using pointer = T*;

  intrusive_ptr() : ptr_(nullptr) {}
  intrusive_ptr(T* ptr) : ptr_(std::shared_ptr<T>(ptr)) {}  // NOLINT
  intrusive_ptr(std::shared_ptr<T> ptr) : ptr_(ptr) {}      // NOLINT

  template <typename... Args>
  static intrusive_ptr<T> make(Args&&... args) {
    return intrusive_ptr<T>(std::make_shared<T>(std::forward<Args>(args)...));
  }

  T* get() const { return ptr_.get(); }
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_.get(); }

  // For IValue
  std::shared_ptr<T> get_shared() const { return ptr_; }

  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  std::shared_ptr<T> ptr_;
};

template <typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
  return intrusive_ptr<T>::make(std::forward<Args>(args)...);
}

enum class TypeTag { None = 0, Bool, Int, Double, String, Tensor, CustomClass };

class IValue {
 private:
  struct CustomClassWrapper {
    std::shared_ptr<CustomClassHolder> ptr;
    std::string class_name;

    CustomClassWrapper(std::shared_ptr<CustomClassHolder> p,
                       const std::string& name)
        : ptr(std::move(p)), class_name(name) {}
  };

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
  template <typename T>
  IValue(intrusive_ptr<T> ptr)  // NOLINT
      : tag_(TypeTag::CustomClass),
        value_(CustomClassWrapper(ptr.get_shared(), typeid(T).name())) {}

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
  bool is_custom_class() const { return tag_ == TypeTag::CustomClass; }

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

  template <typename T>
  intrusive_ptr<T> to_custom_class() const {
    if (!is_custom_class()) throw std::runtime_error("Not a custom class");
    const auto& wrapper = std::get<CustomClassWrapper>(value_);
    auto typed_ptr = std::dynamic_pointer_cast<T>(wrapper.ptr);
    if (!typed_ptr) {
      throw std::runtime_error("Custom class type mismatch");
    }
    return intrusive_ptr<T>(typed_ptr);
  }

  template <typename T>
  auto get() const -> std::conditional_t<
      std::is_same_v<T, std::remove_cv_t<std::remove_reference_t<T>>>,
      T,
      T> {
    using BaseT = std::remove_cv_t<std::remove_reference_t<T>>;

    if constexpr (is_intrusive_ptr_v<BaseT>) {
      using ElementType = typename BaseT::element_type;
      return to_custom_class<ElementType>();
    } else if constexpr (std::is_same_v<BaseT, IValue>) {
      return *this;
    } else {
      BaseT result;
      if (try_convert_to<BaseT>(result)) {
        return result;
      }
      std::ostringstream oss;
      oss << "Cannot convert " << type_string() << " to " << typeid(T).name();
      throw std::runtime_error(oss.str());
    }
  }

 private:
  template <typename T>
  struct is_intrusive_ptr : std::false_type {};

  template <typename T>
  struct is_intrusive_ptr<intrusive_ptr<T>> : std::true_type {};

  template <typename T>
  static constexpr bool is_intrusive_ptr_v = is_intrusive_ptr<T>::value;

 public:
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

  bool try_to_custom_class(std::shared_ptr<CustomClassHolder>& out,  // NOLINT
                           const std::string& expected_class_name) const {
    if (is_custom_class()) {
      const auto& wrapper = std::get<CustomClassWrapper>(value_);
      if (wrapper.class_name == expected_class_name) {
        out = wrapper.ptr;
        return true;
      }
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
    } else if constexpr (is_intrusive_ptr_v<BaseType>) {
      using ElementType = typename BaseType::element_type;
      std::shared_ptr<CustomClassHolder> base_ptr;
      if (try_to_custom_class(base_ptr, typeid(ElementType).name())) {
        auto typed_ptr = std::dynamic_pointer_cast<ElementType>(base_ptr);
        if (typed_ptr) {
          out = intrusive_ptr<ElementType>(typed_ptr);
          return true;
        }
      }
      return false;
    } else {
      return false;
    }
  }

  std::string get_custom_class_name() const {
    if (!is_custom_class()) throw std::runtime_error("Not a custom class");
    const auto& wrapper = std::get<CustomClassWrapper>(value_);
    return wrapper.class_name;
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
      case TypeTag::CustomClass:
        return "CustomClass(" + get_custom_class_name() + ")";
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
      case TypeTag::Tensor: {
        const auto& tensor = std::get<at::Tensor>(value_);
        return "Tensor(" + std::to_string(tensor.numel()) + " elements)";
      }
      case TypeTag::CustomClass: {
        const auto& wrapper = std::get<CustomClassWrapper>(value_);
        return "CustomClass(" + wrapper.class_name + ")";
      }
      default:
        return "Unknown";
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const IValue& val) {
    return os << val.to_repr();
  }

 private:
  TypeTag tag_;
  std::variant<std::monostate,
               bool,
               int,
               double,
               std::string,
               at::Tensor,
               CustomClassWrapper>
      value_;
};

}  // namespace torch
