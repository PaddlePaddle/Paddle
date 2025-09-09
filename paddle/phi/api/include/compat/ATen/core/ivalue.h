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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once
#include <ATen/core/TensorBody.h>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

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

template <typename T>
struct _fake_type {};

enum class TypeTag {
  None = 0,
  Bool,
  Int,
  Double,
  String,
  Tensor,
  GenericList,
  CustomClass,
  Tuple
};

class IValue;  // Forward declaration

// Forward declaration of generic_to template function
template <typename T>
T generic_to(const IValue& ivalue, _fake_type<T>);

using GenericList = std::vector<IValue>;

// Separate tuple wrapper to avoid ambiguity with GenericList
struct GenericTuple {
  std::vector<IValue> elements;

  GenericTuple() = default;
  GenericTuple(std::vector<IValue> elems)  // NOLINT
      : elements(std::move(elems)) {}

  size_t size() const { return elements.size(); }
  IValue& operator[](size_t idx) { return elements[idx]; }
  const IValue& operator[](size_t idx) const { return elements[idx]; }
};

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
  IValue() : tag_(TypeTag::None), value_(std::monostate{}) {}

  IValue(bool val) : tag_(TypeTag::Bool), value_(val) {}  // NOLINT
  IValue(int val)                                         // NOLINT
      : tag_(TypeTag::Int), value_(static_cast<int64_t>(val)) {}
  IValue(int64_t val) : tag_(TypeTag::Int), value_(val) {}    // NOLINT
  IValue(double val) : tag_(TypeTag::Double), value_(val) {}  // NOLINT
  IValue(const std::string& val)                              // NOLINT
      : tag_(TypeTag::String), value_(val) {}
  IValue(std::string&& val)  // NOLINT
      : tag_(TypeTag::String), value_(std::move(val)) {}
  IValue(const char* val)  // NOLINT
      : tag_(TypeTag::String), value_(std::string(val)) {}
  IValue(at::Tensor val) : tag_(TypeTag::Tensor), value_(val) {}  // NOLINT
  IValue(ScalarType val)                                          // NOLINT
      : tag_(TypeTag::Int),
        value_(static_cast<int64_t>(
            static_cast<std::underlying_type_t<ScalarType>>(val))) {}
  template <typename T>
  IValue(intrusive_ptr<T> ptr)  // NOLINT
      : tag_(TypeTag::CustomClass),
        value_(CustomClassWrapper{ptr.get_shared(), typeid(T).name()}) {}

  template <typename T,
            typename = std::enable_if_t<std::is_constructible_v<IValue, T>>>
  IValue(const std::vector<T>& vec)  // NOLINT
      : tag_(TypeTag::GenericList) {
    GenericList generic_list;
    generic_list.reserve(vec.size());
    for (const auto& item : vec) {
      generic_list.emplace_back(IValue(item));
    }
    value_ = std::move(generic_list);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_constructible_v<IValue, T>>>
  IValue(std::vector<T>&& vec)  // NOLINT
      : tag_(TypeTag::GenericList) {
    GenericList generic_list;
    generic_list.reserve(vec.size());
    for (auto&& item : vec) {
      generic_list.emplace_back(IValue(std::move(item)));
    }
    value_ = std::move(generic_list);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_constructible_v<IValue, T>>>
  IValue(ArrayRef<T> arr) : IValue(arr.vec()) {}  // NOLINT

  template <typename T>
  IValue(const std::optional<T>& opt) {  // NOLINT
    if (opt.has_value()) {
      *this = IValue(*opt);
    } else {
      tag_ = TypeTag::None;
      value_ = std::monostate{};
    }
  }

  template <typename T>
  IValue(std::optional<T>&& opt) {  // NOLINT
    if (opt.has_value()) {
      *this = IValue(std::move(*opt));
    } else {
      tag_ = TypeTag::None;
      value_ = std::monostate{};
    }
  }

  // Variadic template constructor for tuple of any number of tensors or
  // IValue-convertible types
  template <typename... Args>
  IValue(const std::tuple<Args...>& tuple_val)  // NOLINT
      : tag_(TypeTag::Tuple) {
    static_assert(sizeof...(Args) > 0, "Tuple must have at least one element");
    std::vector<IValue> elements;
    elements.reserve(sizeof...(Args));
    tuple_to_ivalue_vector(
        tuple_val, elements, std::index_sequence_for<Args...>{});
    value_ = GenericTuple(std::move(elements));
  }

  // Helper function to convert tuple elements to IValue vector using index
  // sequence
  template <typename Tuple, std::size_t... I>
  void tuple_to_ivalue_vector(const Tuple& tuple_val,
                              std::vector<IValue>& elements,  // NOLINT
                              std::index_sequence<I...>) {
    (elements.emplace_back(std::get<I>(tuple_val)), ...);
  }

  IValue(const IValue& other) = default;
  IValue(IValue&& other) = default;
  IValue& operator=(const IValue& other) = default;
  IValue& operator=(IValue&& other) = default;

  bool is_none() const { return tag_ == TypeTag::None; }
  bool is_bool() const { return tag_ == TypeTag::Bool; }
  bool is_int() const { return tag_ == TypeTag::Int; }
  bool is_double() const { return tag_ == TypeTag::Double; }
  bool is_string() const { return tag_ == TypeTag::String; }
  bool is_list() const { return tag_ == TypeTag::GenericList; }
  bool is_tensor() const { return tag_ == TypeTag::Tensor; }
  bool is_custom_class() const { return tag_ == TypeTag::CustomClass; }
  bool is_tuple() const { return tag_ == TypeTag::Tuple; }

  bool to_bool() const {
    if (!is_bool()) throw std::runtime_error("Not a bool");
    return std::get<bool>(value_);
  }

  int64_t to_int() const {
    if (!is_int()) throw std::runtime_error("Not an int");
    return std::get<int64_t>(value_);
  }

  double to_double() const {
    if (!is_double()) throw std::runtime_error("Not a double");
    return std::get<double>(value_);
  }

  const std::string& to_string() const {
    if (!is_string()) throw std::runtime_error("Not a string");
    return std::get<std::string>(value_);
  }

  const GenericList& to_list() const {
    if (!is_list()) throw std::runtime_error("Not a list");
    return std::get<GenericList>(value_);
  }

  GenericList& to_list() {
    if (!is_list()) throw std::runtime_error("Not a list");
    return std::get<GenericList>(value_);
  }

  at::Tensor to_tensor() const {
    if (!is_tensor()) throw std::runtime_error("Not a tensor");
    return std::get<at::Tensor>(value_);
  }

  const GenericTuple& to_tuple() const {
    if (!is_tuple()) throw std::runtime_error("Not a tuple");
    return std::get<GenericTuple>(value_);
  }

  GenericTuple& to_tuple() {
    if (!is_tuple()) throw std::runtime_error("Not a tuple");
    return std::get<GenericTuple>(value_);
  }

  at::ScalarType to_scalar_type() const {
    if (!is_int()) throw std::runtime_error("Not an int");
    return static_cast<at::ScalarType>(std::get<int64_t>(value_));
  }

  template <typename T>
  intrusive_ptr<T> to_custom_class() const {
    if (!is_custom_class()) throw std::runtime_error("Not a custom class");
    const auto& wrapper = std::get<CustomClassWrapper>(value_);
    auto casted = std::dynamic_pointer_cast<T>(wrapper.ptr);
    if (!casted) {
      throw std::runtime_error("Cannot cast custom class to requested type");
    }
    return intrusive_ptr<T>(casted);
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
      out = (std::get<int64_t>(value_) != 0);
      return true;
    } else if (is_double()) {
      out = (std::get<double>(value_) != 0.0);
      return true;
    }
    return false;
  }

  bool try_to_int(int& out) const {  // NOLINT
    if (is_int()) {
      out = static_cast<int>(std::get<int64_t>(value_));
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
      out = static_cast<double>(std::get<int64_t>(value_));
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

  bool try_to_scalar_type(at::ScalarType& out) const {  // NOLINT
    if (is_int()) {
      out = static_cast<at::ScalarType>(std::get<int64_t>(value_));
      return true;
    }
    return false;
  }

  template <typename T>
  bool try_to_optional_type(std::optional<T>& out) const {  // NOLINT
    if (is_none()) {
      out = std::nullopt;
      return true;
    } else {
      T value;
      if (try_convert_to<T>(value)) {
        out = value;
        return true;
      }
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
    } else if constexpr (std::is_same_v<BaseType, at::ScalarType>) {
      return try_to_scalar_type(const_cast<at::ScalarType&>(
          reinterpret_cast<const at::ScalarType&>(out)));
    } else {
      try {
        // Handle const types by removing const and using const_cast
        using NonConstType = std::remove_const_t<T>;
        NonConstType temp = this->to<BaseType>();
        const_cast<NonConstType&>(out) = std::move(temp);
        return true;
      } catch (const std::exception&) {
        return false;
      }
    }
  }

  std::string get_custom_class_name() const {
    if (!is_custom_class()) throw std::runtime_error("Not a custom class");
    const auto& wrapper = std::get<CustomClassWrapper>(value_);
    return wrapper.class_name;
  }

  template <typename T>
  T to() && {
    return generic_to(std::move(*this), _fake_type<T>{});
  }

  template <typename T>
  T to() const& {
    return generic_to(*this, _fake_type<T>{});
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
      case TypeTag::GenericList:
        return "List";
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
        return std::to_string(std::get<int64_t>(value_));
      case TypeTag::Double:
        return std::to_string(std::get<double>(value_));
      case TypeTag::String:
        return "\"" + std::get<std::string>(value_) + "\"";
      case TypeTag::Tensor: {
        const auto& tensor = std::get<at::Tensor>(value_);
        return "Tensor(" + std::to_string(tensor.numel()) + " elements)";
      }
      case TypeTag::GenericList: {
        const auto& list = std::get<GenericList>(value_);
        std::string result = "[";
        for (size_t i = 0; i < list.size(); ++i) {
          if (i > 0) result += ", ";
          result += list[i].to_repr();
        }
        result += "]";
        return result;
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
               int64_t,
               double,
               std::string,
               at::Tensor,
               GenericList,
               CustomClassWrapper,
               GenericTuple>
      value_;
  template <typename T>
  friend T generic_to(const IValue& ivalue, _fake_type<T>);
};

template <>
inline bool generic_to(const IValue& ivalue, _fake_type<bool>) {
  return ivalue.to_bool();
}

template <>
inline int generic_to(const IValue& ivalue, _fake_type<int>) {
  return static_cast<int>(ivalue.to_int());
}

template <>
inline int64_t generic_to(const IValue& ivalue, _fake_type<int64_t>) {
  return ivalue.to_int();
}

template <>
inline double generic_to(const IValue& ivalue, _fake_type<double>) {
  return ivalue.to_double();
}

template <>
inline std::string generic_to(const IValue& ivalue, _fake_type<std::string>) {
  return ivalue.to_string();
}

template <>
inline at::Tensor generic_to(const IValue& ivalue, _fake_type<at::Tensor>) {
  return ivalue.to_tensor();
}

template <typename T>
std::vector<T> generic_to(const IValue& ivalue, _fake_type<std::vector<T>>) {
  auto list = ivalue.to_list();
  std::vector<T> result;
  result.reserve(list.size());
  for (const auto& item : list) {
    result.push_back(item.to<T>());
  }
  return result;
}

template <typename T>
ArrayRef<T> generic_to(const IValue& ivalue, _fake_type<ArrayRef<T>>) {
  static thread_local std::vector<T> temp_storage;
  temp_storage = ivalue.to<std::vector<T>>();
  return ArrayRef<T>(temp_storage);
}

template <typename T>
std::optional<T> generic_to(const IValue& ivalue,
                            _fake_type<std::optional<T>>) {
  if (ivalue.is_none()) {
    return std::nullopt;
  }
  return std::optional<T>(ivalue.to<T>());
}

template <typename T>
intrusive_ptr<T> generic_to(const IValue& ivalue,
                            _fake_type<intrusive_ptr<T>>) {
  return ivalue.to_custom_class<T>();
}

}  // namespace torch
