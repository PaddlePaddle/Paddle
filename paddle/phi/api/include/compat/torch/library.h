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

#include <ATen/core/ivalue.h>

#include <c10/macros/Macros.h>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace torch {
class Library;
class FunctionArgs;
class FunctionResult;

struct arg {
  explicit arg(std::string name)
      : name_(std::move(name)), value_(std::nullopt) {}

  arg& operator=(const IValue& rhs) {
    value_ = rhs;
    return *this;
  }

  static IValue none() { return IValue(); }

  std::string name_;
  std::optional<IValue> value_;
};

template <class... Types>
struct types {
  using type = types;
};

template <class... Types>
struct init_types {
  using type = init_types;
};

template <class... Types>
init_types<Types...> init() {
  return init_types<Types...>{};
}

class FunctionArgs {
 public:
  FunctionArgs() = default;

  template <typename... Args>
  FunctionArgs(Args&&... args) {  // NOLINT
    (add_arg(std::forward<Args>(args)), ...);
  }

  static FunctionArgs from_vector(const std::vector<torch::IValue>& args_vec) {
    FunctionArgs args;
    args.args_ = args_vec;
    return args;
  }

  template <typename T>
  void add_arg(T&& arg) {
    if constexpr (std::is_same_v<std::decay_t<T>, const char*> ||
                  (std::is_array_v<std::decay_t<T>> &&
                   std::is_same_v<std::remove_extent_t<std::decay_t<T>>,
                                  char>)) {
      args_.emplace_back(torch::IValue(std::string(arg)));
    } else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
      args_.emplace_back(torch::IValue(std::forward<T>(arg)));
    } else if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
      args_.emplace_back(torch::IValue(std::forward<T>(arg)));
    } else if constexpr (std::is_same_v<std::decay_t<T>, torch::IValue>) {
      args_.emplace_back(std::forward<T>(arg));
    } else {
      args_.emplace_back(torch::IValue(std::forward<T>(arg)));
    }
  }

  template <typename T>
  auto get(size_t index) const -> std::
      conditional_t<std::is_reference_v<T>, std::remove_reference_t<T>, T> {
    if (index >= args_.size()) {
      throw std::out_of_range("Argument index out of range");
    }

    const torch::IValue& arg = args_[index];

    using ReturnType = std::
        conditional_t<std::is_reference_v<T>, std::remove_reference_t<T>, T>;

    // Handle const references by creating a temporary object
    if constexpr (std::is_const_v<std::remove_reference_t<T>> &&
                  std::is_reference_v<T>) {
      using NonConstType = std::remove_const_t<std::remove_reference_t<T>>;
      NonConstType temp_result;
      if (arg.template try_convert_to<NonConstType>(temp_result)) {
        return temp_result;
      }
    } else if constexpr (std::is_const_v<std::remove_reference_t<ReturnType>>) {
      // Handle const types by using underlying non-const type for conversion
      using NonConstType = std::remove_const_t<ReturnType>;
      NonConstType temp_result;
      if (arg.template try_convert_to<NonConstType>(temp_result)) {
        return static_cast<ReturnType>(temp_result);
      }
    } else {
      ReturnType result;
      if (arg.template try_convert_to<ReturnType>(result)) {
        return result;
      }
    }

    std::ostringstream oss;
    oss << "Cannot convert argument " << index << " from " << arg.type_string()
        << " to " << typeid(T).name();
    throw std::runtime_error(oss.str());
  }

  // Convert to a tuple of specified types
  template <typename... Types>
  std::tuple<Types...> to_tuple() const {
    if (sizeof...(Types) != args_.size()) {
      throw std::runtime_error("Argument count mismatch: expected " +
                               std::to_string(sizeof...(Types)) + ", got " +
                               std::to_string(args_.size()));
    }
    return to_tuple_impl<Types...>(
        std::make_index_sequence<sizeof...(Types)>{});
  }

  size_t size() const { return args_.size(); }

  bool empty() const { return args_.empty(); }

  const IValue& operator[](size_t index) const { return args_[index]; }
  IValue& operator[](size_t index) { return args_[index]; }

  const torch::IValue& get_value(size_t index) const {
    if (index >= args_.size()) {
      throw std::out_of_range("Argument index out of range");
    }
    return args_[index];
  }

  auto begin() const { return args_.begin(); }
  auto end() const { return args_.end(); }

  std::string to_string() const {
    std::ostringstream oss;
    oss << "FunctionArgs[";
    for (size_t i = 0; i < args_.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << args_[i];
    }
    oss << "]";
    return oss.str();
  }

 private:
  template <typename... Types, size_t... I>
  std::tuple<Types...> to_tuple_impl(std::index_sequence<I...>) const {
    return std::make_tuple(get<Types>(I)...);
  }
  std::vector<torch::IValue> args_;
};

class FunctionResult {
 public:
  FunctionResult() : value_(torch::IValue()) {}

  template <typename T>
  FunctionResult(T&& value)  // NOLINT
      : value_(torch::IValue(std::forward<T>(value))) {}

  FunctionResult(const torch::IValue& value) : value_(value) {}        // NOLINT
  FunctionResult(torch::IValue&& value) : value_(std::move(value)) {}  // NOLINT

  template <typename T>
  T get() const {
    if (value_.is_none()) {
      throw std::runtime_error("No return value (void function)");
    }

    T result;
    if (value_.try_convert_to<T>(result)) {
      return result;
    }

    throw std::runtime_error("Cannot convert result from " +
                             value_.type_string() + " to " + typeid(T).name());
  }

  bool has_value() const { return !value_.is_none(); }

  const torch::IValue& get_value() const { return value_; }

  static FunctionResult void_result() { return FunctionResult(); }

  std::string to_string() const {
    return "FunctionResult(" + value_.to_repr() + ")";
  }

 private:
  torch::IValue value_;
};

template <typename T>
struct function_traits;

// Basic function type
template <typename R, typename... Args>
struct function_traits<R(Args...)> {
  using return_type = R;
  static constexpr size_t arity = sizeof...(Args);
  using ArgsTuple = std::tuple<Args...>;

  template <size_t i>
  struct arg {
    using type = typename std::tuple_element<i, std::tuple<Args...>>::type;
  };

  // Generic function call interface
  template <typename F>
  static IValue call_function(F&& func, const FunctionArgs& args) {
    if (args.size() != sizeof...(Args)) {
      throw std::runtime_error(
          "Function expects " + std::to_string(sizeof...(Args)) +
          " arguments, got " + std::to_string(args.size()));
    }
    return call_function_impl(std::forward<F>(func),
                              args,
                              std::make_index_sequence<sizeof...(Args)>{});
  }

 private:
  template <typename F, size_t... I>
  static IValue call_function_impl(F&& func,
                                   const FunctionArgs& args,
                                   std::index_sequence<I...>) {
    auto args_without_ref =
        std::make_tuple(args.template get<std::remove_reference_t<Args>>(I)...);
    if constexpr (std::is_void_v<R>) {
      func(std::get<I>(args_without_ref)...);
      return IValue();
    } else {
      auto result = func(std::get<I>(args_without_ref)...);
      return IValue(result);
    }
  }
};

// Function pointer specialization
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> : public function_traits<R(Args...)> {};

// Reference to function type specialization
template <typename R, typename... Args>
struct function_traits<R (&)(Args...)> : public function_traits<R(Args...)> {};

// Const function type specialization
template <typename R, typename... Args>
struct function_traits<R(Args...) const> : public function_traits<R(Args...)> {
};

// Const function pointer specialization
template <typename R, typename... Args>
struct function_traits<R (*const)(Args...)>
    : public function_traits<R(Args...)> {};

// Common Reference and Pointer types
template <typename T>
struct function_traits<T&>
    : public function_traits<std::remove_reference_t<T>> {};

template <typename T>
struct function_traits<T*> : public function_traits<T> {};

// Member function pointer specialization
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)>
    : public function_traits<R(C&, Args...)> {
  using class_type = C;

  static IValue call_method(R (C::*func)(Args...),
                            C* instance,
                            const FunctionArgs& args) {
    if (args.size() != sizeof...(Args) + 1) {  // +1 for this pointer
      throw std::runtime_error(
          "Method expects " + std::to_string(sizeof...(Args)) +
          " arguments (plus this), got " + std::to_string(args.size() - 1));
    }
    return call_method_impl(
        func, instance, args, std::make_index_sequence<sizeof...(Args)>{});
  }

 private:
  template <size_t... I>
  static IValue call_method_impl(R (C::*func)(Args...),
                                 C* instance,
                                 const FunctionArgs& args,
                                 std::index_sequence<I...>) {
    // Skip args[0] which is 'this'
    auto args_without_ref = std::make_tuple(
        args.template get<std::remove_reference_t<Args>>(I + 1)...);
    if constexpr (std::is_void_v<R>) {
      (instance->*func)(std::get<I>(args_without_ref)...);
      return IValue();
    } else {
      auto result = (instance->*func)(std::get<I>(args_without_ref)...);
      return IValue(result);
    }
  }
};

// Const member function pointer specialization
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const>
    : public function_traits<R(const C&, Args...)> {
  using class_type = C;

  static IValue call_method(R (C::*func)(Args...) const,
                            C* instance,
                            const FunctionArgs& args) {
    if (args.size() != sizeof...(Args) + 1) {  // +1 for this pointer
      throw std::runtime_error(
          "Method expects " + std::to_string(sizeof...(Args)) +
          " arguments (plus this), got " + std::to_string(args.size() - 1));
    }
    return call_method_impl(
        func, instance, args, std::make_index_sequence<sizeof...(Args)>{});
  }

 private:
  template <size_t... I>
  static IValue call_method_impl(R (C::*func)(Args...) const,
                                 C* instance,
                                 const FunctionArgs& args,
                                 std::index_sequence<I...>) {
    if constexpr (std::is_void_v<R>) {
      (instance->*func)(
          args.get<Args>(I + 1)...);  // Skip args[0] which is 'this'
      return IValue();
    } else {
      auto result = (instance->*func)(args.get<Args>(I + 1)...);
      return IValue(result);
    }
  }
};

template <typename Func>
IValue invoke_function(Func&& func, const FunctionArgs& args) {
  using traits =
      function_traits<std::remove_cv_t<std::remove_reference_t<Func>>>;
  return traits::call_function(std::forward<Func>(func), args);
}

template <typename Func, typename Class>
IValue invoke_member_function(Func&& func,
                              Class* instance,
                              const FunctionArgs& args) {
  using traits =
      function_traits<std::remove_cv_t<std::remove_reference_t<Func>>>;
  return traits::call_method(func, instance, args);
}

class CppFunction {
 public:
  using CallableFunction = std::function<FunctionResult(const FunctionArgs&)>;

  CppFunction() : func_(nullptr) {}

  // Constructor for lambda or function object
  explicit CppFunction(std::function<IValue(const FunctionArgs&)> func)
      : func_([func](const FunctionArgs& args) -> FunctionResult {
          try {
            auto result = func(args);
            return FunctionResult(result);
          } catch (const std::exception& e) {
            throw std::runtime_error("Constructor failed: " +
                                     std::string(e.what()));
          }
        }) {}

  // Common function pointer or member function pointer constructor
  template <typename Func>
  explicit CppFunction(
      Func&& f,
      typename std::enable_if_t<
          std::is_function_v<std::remove_pointer_t<std::decay_t<Func>>> ||
          (std::is_pointer_v<std::decay_t<Func>> &&
           std::is_function_v<std::remove_pointer_t<std::decay_t<Func>>>)>* =
          nullptr)
      : func_([f = std::forward<Func>(f)](
                  const FunctionArgs& args) -> FunctionResult {
          try {
            auto result = invoke_function(f, args);
            return FunctionResult(result);
          } catch (const std::exception& e) {
            throw std::runtime_error("Function call failed: " +
                                     std::string(e.what()));
          }
        }) {}

  // Common member function pointer constructor
  template <typename Func>
  explicit CppFunction(
      Func&& f,
      typename std::enable_if_t<
          !std::is_function_v<std::remove_pointer_t<std::decay_t<Func>>> &&
          !std::is_pointer_v<std::decay_t<Func>> &&
          std::is_invocable_v<Func, const FunctionArgs&>>* = nullptr)
      : func_([f = std::forward<Func>(f)](
                  const FunctionArgs& args) -> FunctionResult {
          try {
            auto result = f(args);
            return FunctionResult(result);
          } catch (const std::exception& e) {
            throw std::runtime_error("Lambda execution failed: " +
                                     std::string(e.what()));
          }
        }) {}

  CppFunction(CppFunction&& other) noexcept : func_(std::move(other.func_)) {}

  CppFunction& operator=(CppFunction&& other) noexcept {
    if (this != &other) {
      func_ = std::move(other.func_);
    }
    return *this;
  }

  CppFunction(const CppFunction&) = delete;
  CppFunction& operator=(const CppFunction&) = delete;

  FunctionResult call() const {
    if (!func_) {
      throw std::runtime_error("CppFunction is not initialized");
    }
    return func_(FunctionArgs{});
  }

  template <typename... Args>
  FunctionResult call(Args&&... args) const {
    if (!func_) {
      throw std::runtime_error("CppFunction is not initialized");
    }
    return func_(FunctionArgs{std::forward<Args>(args)...});
  }

  FunctionResult call_with_args(const FunctionArgs& args) const {
    if (!func_) {
      throw std::runtime_error("CppFunction is not initialized");
    }
    return func_(args);
  }

  bool valid() const { return func_ != nullptr; }

 private:
  CallableFunction func_;
};

struct ClassRegistration {
  std::string namespace_name;
  std::string class_name;
  std::string qualified_name;
  std::vector<std::shared_ptr<CppFunction>> constructors;
  std::unordered_map<std::string, std::shared_ptr<CppFunction>> methods;
  std::unordered_map<std::string, std::shared_ptr<CppFunction>> static_methods;

  ClassRegistration() = default;
  ClassRegistration(const std::string& ns, const std::string& name)
      : namespace_name(ns),
        class_name(name),
        qualified_name(ns + "::" + name) {}
};

// Global class registry
class ClassRegistry {
 public:
  static ClassRegistry& instance() {
    static ClassRegistry registry;
    return registry;
  }

  void register_class(const std::string& namespace_name,
                      const std::string& class_name) {
    std::string qualified_name = namespace_name + "::" + class_name;
    classes_[qualified_name] =
        std::make_unique<ClassRegistration>(namespace_name, class_name);
    // TODO(SigureMo): Use vlog for debug logging
    // std::cout << "Registered class: " << qualified_name << std::endl;
  }

  void register_constructor(const std::string& qualified_name,
                            CppFunction&& func) {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) {
      throw std::runtime_error("Class " + qualified_name + " not found");
    }
    it->second->constructors.push_back(
        std::make_shared<CppFunction>(std::move(func)));
    // std::cout << "Registered constructor for: " << qualified_name
    //           << " (total: " << it->second->constructors.size() << ")"
    //           << std::endl;
  }

  void register_method(const std::string& qualified_name,
                       const std::string& method_name,
                       CppFunction&& func) {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) {
      throw std::runtime_error("Class " + qualified_name + " not found");
    }
    it->second->methods[method_name] =
        std::make_shared<CppFunction>(std::move(func));
    // std::cout << "Registered method: " << qualified_name << "::" <<
    // method_name
    //           << std::endl;
  }

  void register_static_method(const std::string& qualified_name,
                              const std::string& method_name,
                              CppFunction&& func) {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) {
      throw std::runtime_error("Class " + qualified_name + " not found");
    }
    it->second->static_methods[method_name] =
        std::make_shared<CppFunction>(std::move(func));
    // std::cout << "Registered static method: " << qualified_name
    //           << "::" << method_name << std::endl;
  }

  bool has_class(const std::string& qualified_name) const {
    return classes_.find(qualified_name) != classes_.end();
  }

  bool has_method(const std::string& qualified_name,
                  const std::string& method_name) const {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) return false;
    return it->second->methods.find(method_name) != it->second->methods.end();
  }

  bool has_static_method(const std::string& qualified_name,
                         const std::string& method_name) const {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) return false;
    return it->second->static_methods.find(method_name) !=
           it->second->static_methods.end();
  }

  FunctionResult call_method_with_args(const std::string& qualified_name,
                                       const std::string& method_name,
                                       const FunctionArgs& args) {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) {
      throw std::runtime_error("Class " + qualified_name + " not found!");
    }

    auto& class_reg = it->second;
    auto method_it = class_reg->methods.find(method_name);
    if (method_it == class_reg->methods.end()) {
      throw std::runtime_error("Method " + method_name + " not found in " +
                               qualified_name + "!");
    }

    try {
      // std::cout << "Executing " << qualified_name << "::" << method_name
      //           << " (instance) with " << args.size() << " args" <<
      //           std::endl;
      auto result = method_it->second->call_with_args(args);

      if (result.has_value()) {
        // std::cout << "Instance method executed successfully with return
        // value"
        //           << std::endl;
      } else {
        // std::cout << "Instance method executed successfully (void)"
        //           << std::endl;
      }
      return result;
    } catch (const std::exception& e) {
      // std::cout << "Instance method execution failed: " << e.what()
      //           << std::endl;
      throw;
    }
  }

  FunctionResult call_constructor_with_args(const std::string& qualified_name,
                                            const FunctionArgs& args) const {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) {
      throw std::runtime_error("Class " + qualified_name + " not found!");
    }

    auto& class_reg = it->second;
    if (class_reg->constructors.empty()) {
      throw std::runtime_error("No constructor registered for " +
                               qualified_name);
    }

    // std::cout << "Creating instance of " << qualified_name << " with "
    //           << args.size() << " args" << std::endl;
    // std::cout << "Available constructors: " << class_reg->constructors.size()
    //           << std::endl;

    for (size_t i = 0; i < class_reg->constructors.size(); ++i) {
      try {
        // std::cout << "Trying constructor " << (i + 1) << "..." << std::endl;
        auto result = class_reg->constructors[i]->call_with_args(args);
        // std::cout << "Constructor " << (i + 1) << " executed successfully"
        //           << std::endl;
        return result;
      } catch (const std::exception& e) {
        // std::cout << "Constructor " << (i + 1) << " failed: " << e.what()
        //           << std::endl;
      }
    }

    throw std::runtime_error("No suitable constructor found for " +
                             qualified_name);
  }

  FunctionResult call_static_method_with_args(const std::string& qualified_name,
                                              const std::string& method_name,
                                              const FunctionArgs& args) const {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) {
      throw std::runtime_error("Class " + qualified_name + " not found!");
    }

    auto& class_reg = it->second;
    auto method_it = class_reg->static_methods.find(method_name);
    if (method_it == class_reg->static_methods.end()) {
      throw std::runtime_error("Static method " + method_name +
                               " not found in " + qualified_name + "!");
    }

    try {
      // std::cout << "Executing " << qualified_name << "::" << method_name
      //           << " (static) with " << args.size() << " args" << std::endl;
      auto result = method_it->second->call_with_args(args);

      if (result.has_value()) {
        // std::cout << "Static method executed successfully with return value"
        //           << std::endl;
      } else {
        // std::cout << "Static method executed successfully (void return)"
        //           << std::endl;
      }
      return result;
    } catch (const std::exception& e) {
      // std::cout << "Error executing static method: " << e.what() <<
      // std::endl;
      throw;
    }
  }

  FunctionResult call_method_with_args(const std::string& qualified_name,
                                       const std::string& method_name,
                                       const IValue& instance,
                                       const FunctionArgs& args) const {
    auto it = classes_.find(qualified_name);
    if (it == classes_.end()) {
      throw std::runtime_error("Class " + qualified_name + " not found!");
    }

    auto& class_reg = it->second;
    auto method_it = class_reg->methods.find(method_name);
    if (method_it == class_reg->methods.end()) {
      throw std::runtime_error("Instance method " + method_name +
                               " not found in " + qualified_name + "!");
    }

    try {
      // std::cout << "Executing " << qualified_name << "::" << method_name
      //           << " (instance) with " << args.size() << " args" <<
      //           std::endl;

      // Create a FunctionArgs object with the instance as the first argument
      FunctionArgs method_args;
      method_args.add_arg(instance);  // Add the instance as the first arg
      for (size_t i = 0; i < args.size(); ++i) {
        method_args.add_arg(args.get_value(i));
      }

      auto result = method_it->second->call_with_args(method_args);

      if (result.has_value()) {
        // std::cout << "Instance method executed successfully with return
        // value"
        //           << std::endl;
      } else {
        // std::cout << "Instance method executed successfully (void return)"
        //           << std::endl;
      }
      return result;
    } catch (const std::exception& e) {
      // std::cout << "Error executing instance method: " << e.what() <<
      // std::endl;
      throw;
    }
  }

  void print_all_classes() const {
    std::cout << "\n=== Registered Classes ===" << std::endl;
    for (const auto& [qualified_name, registration] : classes_) {
      std::cout << "Class: " << qualified_name << std::endl;

      if (!registration->constructors.empty()) {
        std::cout << "  Constructors: " << registration->constructors.size()
                  << " available" << std::endl;
      }

      if (!registration->methods.empty()) {
        std::cout << "  Methods: ";
        for (const auto& [method_name, _] : registration->methods) {
          std::cout << method_name << " ";
        }
        std::cout << std::endl;
      }

      if (!registration->static_methods.empty()) {
        std::cout << "  Static Methods: ";
        for (const auto& [method_name, _] : registration->static_methods) {
          std::cout << method_name << " ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << "==========================" << std::endl << std::endl;
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<ClassRegistration>> classes_;
};

// Class registration API
template <class CurClass>
class class_ {
  static_assert(
      std::is_base_of_v<torch::CustomClassHolder, CurClass>,
      "torch::class_<T> requires T to inherit from CustomClassHolder");

 public:
  class_(const std::string& namespaceName, const std::string& className)
      : namespace_name_(namespaceName),
        class_name_(className),
        qualified_name_(namespaceName + "::" + className) {
    ClassRegistry::instance().register_class(namespaceName, className);
  }

  // Register constructor
  template <typename... Types>
  class_& def(torch::init_types<Types...>) {
    // std::cout << "def() called with " << sizeof...(Types)
    //           << " template parameters" << std::endl;

    // Create a lambda for the constructor
    auto constructor_func = [](const FunctionArgs& args) -> torch::IValue {
      // std::cout << "Constructor lambda called with " << args.size()
      //           << " arguments" << std::endl;
      // std::cout << "Expected parameter count: " << sizeof...(Types)
      //           << std::endl;

      if constexpr (sizeof...(Types) == 0) {
        // Default constructor
        if (args.size() != 0) {
          throw std::runtime_error(
              "Default constructor expects 0 arguments, got " +
              std::to_string(args.size()));
        }
        auto instance = torch::make_intrusive<CurClass>();
        return torch::IValue(instance);
      } else {
        // Parameterized constructor
        if (args.size() != sizeof...(Types)) {
          throw std::runtime_error(
              "Constructor argument count mismatch: expected " +
              std::to_string(sizeof...(Types)) + ", got " +
              std::to_string(args.size()));
        }
        // Use std::apply to unpack the arguments
        auto tuple_args = args.to_tuple<Types...>();
        auto instance = std::apply(
            [](Types... args) {
              return torch::make_intrusive<CurClass>(
                  std::forward<Types>(args)...);
            },
            tuple_args);
        return torch::IValue(instance);
      }
    };

    ClassRegistry::instance().register_constructor(
        qualified_name_, CppFunction(constructor_func));
    return *this;
  }

  // Register instance method
  template <typename Func>
  class_& def(const std::string& name, Func&& f) {
    // Check if Func is a member function pointer
    if constexpr (std::is_member_function_pointer_v<std::decay_t<Func>>) {
      // Use function_traits to extract class type and method signature
      auto method_func = [f](const FunctionArgs& args) -> torch::IValue {
        if (args.size() < 1) {
          throw std::runtime_error(
              "Instance method requires at least 1 argument (this pointer)");
        }

        // Get the instance (first argument)
        auto instance = args.get<torch::intrusive_ptr<CurClass>>(0);

        // Invoke the member function
        return invoke_member_function(f, instance.get(), args);
      };

      ClassRegistry::instance().register_method(
          qualified_name_, name, CppFunction(method_func));
      // std::cout << "Instance method " << name << " registered successfully"
      //           << std::endl;
    } else {
      // Handle generic callable (e.g., lambda, std::function)
      // std::cout << "Method registration for " << name
      //           << " (generic callable not yet implemented)" << std::endl;
    }

    return *this;
  }

  // Register static method
  template <typename Func>
  class_& def_static(const std::string& name, Func&& f) {
    ClassRegistry::instance().register_static_method(
        qualified_name_, name, CppFunction(std::forward<Func>(f)));
    return *this;
  }

 private:
  std::string namespace_name_;
  std::string class_name_;
  std::string qualified_name_;
};

enum class DispatchKey {
  Undefined = 0,
  CPU,
  CUDA,
};

inline std::string dispatch_key_to_string(DispatchKey key) {
  switch (key) {
    case DispatchKey::CPU:
      return "CPU";
    case DispatchKey::CUDA:
      return "CUDA";
    default:
      return "Undefined";
  }
}

// Operator Registration
struct OperatorRegistration {
  std::string qualified_name;  // namespace::op_name
  std::string schema;
  std::unordered_map<DispatchKey, CppFunction> implementations;

  OperatorRegistration(const std::string& name,
                       const std::string& schema_str = "")
      : qualified_name(name), schema(schema_str) {}
};

class OperatorRegistry {
 public:
  static OperatorRegistry& instance() {
    static OperatorRegistry registry;
    return registry;
  }

  void register_schema(const std::string& qualified_name,
                       const std::string& schema) {
    auto& op = get_or_create_operator(qualified_name);
    op.schema = schema;
    // std::cout << "Registered schema: " << qualified_name << " -> " << schema
    //           << std::endl;
  }

  void register_implementation(const std::string& qualified_name,
                               DispatchKey key,
                               CppFunction&& func) {
    auto& op = get_or_create_operator(qualified_name);
    op.implementations[key] = std::move(func);
    // std::cout << "Registered implementation: " << qualified_name << " for "
    //           << dispatch_key_to_string(key) << std::endl;
  }

  OperatorRegistration* find_operator(const std::string& qualified_name) {
    auto it = operators_.find(qualified_name);
    return (it != operators_.end()) ? &it->second : nullptr;
  }

  std::vector<std::string> list_all_operators() const {
    std::vector<std::string> ops;
    for (const auto& pair : operators_) {
      ops.push_back(pair.first);
    }
    return ops;
  }

  bool execute_operator(const std::string& qualified_name,
                        DispatchKey key = DispatchKey::CPU) {
    auto* op = find_operator(qualified_name);
    if (!op) {
      // std::cout << "Error: Operator " << qualified_name << " not found!"
      //           << std::endl;
      return false;
    }

    auto impl_it = op->implementations.find(key);
    if (impl_it != op->implementations.end()) {
      try {
        // std::cout << "Executing " << qualified_name << " with "
        //           << dispatch_key_to_string(key) << std::endl;
        auto result = impl_it->second.call();
        if (result.has_value()) {
          // std::cout << "Operator executed successfully with return value"
          //           << std::endl;
        } else {
          // std::cout << "Operator executed successfully (void return)"
          //           << std::endl;
        }
        return true;
      } catch (const std::exception& e) {
        // std::cout << "Error executing operator: " << e.what() << std::endl;
        return false;
      }
    }

    // try fallback to CPU
    if (key != DispatchKey::CPU) {
      auto cpu_it = op->implementations.find(DispatchKey::CPU);
      if (cpu_it != op->implementations.end()) {
        // std::cout << "Fallback to CPU for " << qualified_name << std::endl;
        try {
          auto result = cpu_it->second.call();
          if (result.has_value()) {
            // std::cout << "Operator executed successfully with return value "
            //              "(CPU fallback)"
            //           << std::endl;
          } else {
            // std::cout
            //     << "Operator executed successfully (void return, CPU
            //     fallback)"
            //     << std::endl;
          }
          return true;
        } catch (const std::exception& e) {
          // std::cout << "Error executing operator (CPU fallback): " <<
          // e.what()
          //           << std::endl;
          return false;
        }
      }
    }

    // std::cout << "Error: No implementation found for " << qualified_name
    //           << " with " << dispatch_key_to_string(key) << std::endl;
    return false;
  }

  template <typename... Args>
  FunctionResult execute_operator_with_args(const std::string& qualified_name,
                                            DispatchKey key,
                                            Args&&... args) {
    auto* op = find_operator(qualified_name);
    if (!op) {
      throw std::runtime_error("Operator " + qualified_name + " not found!");
    }

    auto impl_it = op->implementations.find(key);
    if (impl_it != op->implementations.end()) {
      try {
        // std::cout << "Executing " << qualified_name << " with "
        //           << dispatch_key_to_string(key) << std::endl;
        auto result = impl_it->second.call(std::forward<Args>(args)...);
        if (result.has_value()) {
          // std::cout << "Operator executed successfully with return value"
          //           << std::endl;
        } else {
          // std::cout << "Operator executed successfully (void return)"
          //           << std::endl;
        }
        return result;
      } catch (const std::exception& e) {
        throw std::runtime_error("Error executing operator: " +
                                 std::string(e.what()));
      }
    }

    // try fallback to CPU
    if (key != DispatchKey::CPU) {
      auto cpu_it = op->implementations.find(DispatchKey::CPU);
      if (cpu_it != op->implementations.end()) {
        // std::cout << "Fallback to CPU for " << qualified_name << std::endl;
        try {
          auto result = cpu_it->second.call(std::forward<Args>(args)...);
          if (result.has_value()) {
            // std::cout << "Operator executed successfully with return value "
            //              "(CPU fallback)"
            //           << std::endl;
          } else {
            // std::cout
            //     << "Operator executed successfully (void return, CPU
            //     fallback)"
            //     << std::endl;
          }
          return result;
        } catch (const std::exception& e) {
          throw std::runtime_error("Error executing operator (CPU fallback): " +
                                   std::string(e.what()));
        }
      }
    }

    throw std::runtime_error("No implementation found for " + qualified_name +
                             " with " + dispatch_key_to_string(key));
  }

  const std::unordered_map<std::string, OperatorRegistration>& get_operators()
      const {
    return operators_;
  }

  void print_all_operators() const {
    std::cout << "\n=== Registered Operators ===" << std::endl;
    for (const auto& [name, op] : operators_) {
      std::cout << "Operator: " << name << std::endl;
      if (!op.schema.empty()) {
        std::cout << "  Schema: " << op.schema << std::endl;
      }
      std::cout << "  Implementations: ";
      for (const auto& [key, impl] : op.implementations) {
        std::cout << dispatch_key_to_string(key) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "=========================" << std::endl;
  }

 private:
  std::unordered_map<std::string, OperatorRegistration> operators_;

  OperatorRegistration& get_or_create_operator(
      const std::string& qualified_name) {
    auto it = operators_.find(qualified_name);
    if (it == operators_.end()) {
      auto [new_it, inserted] = operators_.emplace(
          qualified_name, OperatorRegistration(qualified_name));
      return new_it->second;
    }
    return it->second;
  }
};

class Library {
 public:
  enum Kind {
    DEF,      // TORCH_LIBRARY
    IMPL,     // TORCH_LIBRARY_IMPL
    FRAGMENT  // TORCH_LIBRARY_FRAGMENT
  };

  Library(Kind kind,
          const std::string& ns,
          std::optional<DispatchKey> dispatch_key = std::nullopt,
          const char* file = nullptr,
          uint32_t line = 0)
      : kind_(kind),
        ns_(ns),
        dispatch_key_(dispatch_key),
        file_(file),
        line_(line) {
    // std::cout << "Created Library: kind=" << kind_to_string(kind)
    //           << ", namespace=" << ns;
    if (dispatch_key) {
      // std::cout << ", dispatch_key=" <<
      // dispatch_key_to_string(*dispatch_key);
    }
    // std::cout << std::endl;
  }

  Library(const std::string& ns)  // NOLINT
      : kind_(DEF), ns_(ns), file_(nullptr), line_(0) {
    // std::cout << "Created Library: namespace=" << ns << std::endl;
  }

  // Define an operator schema (for TORCH_LIBRARY and TORCH_LIBRARY_FRAGMENT)
  Library& def(const std::string& schema) & {
    if (kind_ == IMPL) {
      // std::cout
      //     << "Warning: def() should not be called in TORCH_LIBRARY_IMPL
      //     block"
      //     << std::endl;
      return *this;
    }

    // Simple schema extraction: if it contains '(', extract the part before '('
    auto op_name = extract_op_name(schema);
    auto qualified_name = ns_ + "::" + op_name;

    OperatorRegistry::instance().register_schema(qualified_name, schema);
    return *this;
  }

  // Define an operator implementation
  template <typename Func>
  Library& def(const std::string& name_or_schema, Func&& f) & {
    auto op_name = extract_op_name(name_or_schema);
    auto qualified_name = ns_ + "::" + op_name;

    // If name_or_schema contains '(', treat it as a schema
    if (name_or_schema.find('(') != std::string::npos) {
      OperatorRegistry::instance().register_schema(qualified_name,
                                                   name_or_schema);
    }

    // Register implementation
    auto dispatch_key = dispatch_key_.value_or(DispatchKey::CPU);
    OperatorRegistry::instance().register_implementation(
        qualified_name, dispatch_key, CppFunction(std::forward<Func>(f)));

    return *this;
  }

  // Implementation of an operator
  template <typename Func>
  Library& impl(const std::string& op_name, Func&& f) & {
    auto qualified_name = ns_ + "::" + op_name;
    auto dispatch_key = dispatch_key_.value_or(DispatchKey::CPU);

    OperatorRegistry::instance().register_implementation(
        qualified_name, dispatch_key, CppFunction(std::forward<Func>(f)));

    return *this;
  }

  template <class CurClass>
  ::torch::class_<CurClass> class_(const std::string& className) {
    return ::torch::class_<CurClass>(ns_, className);
  }

  // Print current library info
  void print_info() const {
    // std::cout << "Library Info: " << kind_to_string(kind_)
    //           << ", namespace=" << ns_;
    if (dispatch_key_) {
      // std::cout << ", dispatch_key=" <<
      // dispatch_key_to_string(*dispatch_key_);
    }
    // std::cout << std::endl;
  }

 private:
  Kind kind_;
  std::string ns_;
  std::optional<DispatchKey> dispatch_key_;
  const char* file_;
  uint32_t line_;

  std::string extract_op_name(const std::string& name_or_schema) const {
    // Extract the operator name from the schema string
    auto pos = name_or_schema.find('(');
    if (pos != std::string::npos) {
      return name_or_schema.substr(0, pos);
    }
    return name_or_schema;
  }

  std::string kind_to_string(Kind kind) const {
    switch (kind) {
      case DEF:
        return "DEF";
      case IMPL:
        return "IMPL";
      case FRAGMENT:
        return "FRAGMENT";
      default:
        return "UNKNOWN";
    }
  }
};

namespace detail {

class TorchLibraryInit {
 public:
  using InitFn = void(Library&);

  TorchLibraryInit(Library::Kind kind,
                   InitFn* fn,
                   const char* ns,
                   std::optional<DispatchKey> dispatch_key,
                   const char* file,
                   uint32_t line) {
    Library lib(kind, ns, dispatch_key, file, line);
    fn(lib);
  }
};

}  // namespace detail

// TORCH_LIBRARY
#define TORCH_LIBRARY(ns, m)                                                   \
  static void TORCH_LIBRARY_init_##ns(torch::Library&);                        \
  static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_##ns( \
      torch::Library::DEF,                                                     \
      &TORCH_LIBRARY_init_##ns,                                                \
      #ns,                                                                     \
      std::nullopt,                                                            \
      __FILE__,                                                                \
      __LINE__);                                                               \
  void TORCH_LIBRARY_init_##ns(torch::Library& m)  // NOLINT

// TORCH_LIBRARY_FRAGMENT
#define TORCH_LIBRARY_FRAGMENT(ns, m) _TORCH_LIBRARY_FRAGMENT(ns, m, C10_UID)
#define _TORCH_LIBRARY_FRAGMENT(ns, m, uid)                        \
  static void C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_init_##ns##_, \
                              uid)(torch::Library&);               \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(    \
      TORCH_LIBRARY_FRAGMENT_static_init_##ns##_, uid)(            \
      torch::Library::FRAGMENT,                                    \
      &C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid),  \
      #ns,                                                         \
      std::nullopt,                                                \
      __FILE__,                                                    \
      __LINE__);                                                   \
  void C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_init_##ns##_,        \
                       uid)(torch::Library & m)  // NOLINT

// TORCH_LIBRARY_IMPL
#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)
#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)                           \
  static void C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, \
                              uid)(torch::Library&);                 \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(      \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(            \
      torch::Library::IMPL,                                          \
      &C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid),  \
      #ns,                                                           \
      std::make_optional(torch::DispatchKey::k),                     \
      __FILE__,                                                      \
      __LINE__);                                                     \
  void C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_##ns##_##k##_,        \
                       uid)(torch::Library & m)  // NOLINT

}  // namespace torch
