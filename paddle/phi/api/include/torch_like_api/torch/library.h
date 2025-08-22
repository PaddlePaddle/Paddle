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

#include <ATen/core/ivalue.h>

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

    ReturnType result;
    if (arg.template try_convert_to<ReturnType>(result)) {
      return result;
    }

    std::ostringstream oss;
    oss << "Cannot convert argument " << index << " from " << arg.type_string()
        << " to " << typeid(T).name();
    throw std::runtime_error(oss.str());
  }

  size_t size() const { return args_.size(); }

  bool empty() const { return args_.empty(); }

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

// 普通函数指针
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
  using return_type = R;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename R, typename... Args>
struct function_traits<R (&)(Args...)> {
  using return_type = R;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...)> {
  using return_type = R;
  using args_tuple = std::tuple<C*, Args...>;
  static constexpr size_t arity = sizeof...(Args) + 1;
};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const> {
  using return_type = R;
  using args_tuple = std::tuple<const C*, Args...>;
  static constexpr size_t arity = sizeof...(Args) + 1;
};

template <typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};

template <typename ArgsTuple, typename Func, size_t... I>
auto call_with_args_impl(Func&& f,
                         const FunctionArgs& args,
                         std::index_sequence<I...>) {
  if (args.size() != sizeof...(I)) {
    std::ostringstream oss;
    oss << "Argument count mismatch: expected " << sizeof...(I) << ", got "
        << args.size();
    throw std::runtime_error(oss.str());
  }

  if constexpr (sizeof...(I) == 0) {
    return f();
  } else {
    return f(args.get<std::tuple_element_t<I, ArgsTuple>>(I)...);
  }
}

class CppFunction {
 public:
  using CallableFunction = std::function<FunctionResult(const FunctionArgs&)>;

  CppFunction() : func_(nullptr) {}

  template <typename Func>
  explicit CppFunction(Func&& f) {
    using FuncTraits = function_traits<std::decay_t<Func>>;
    using ReturnType = typename FuncTraits::return_type;
    using ArgsTuple = typename FuncTraits::args_tuple;

    func_ = [f = std::forward<Func>(f)](
                const FunctionArgs& args) -> FunctionResult {
      if constexpr (std::is_void_v<ReturnType>) {
        call_with_args_impl<ArgsTuple>(
            f, args, std::make_index_sequence<FuncTraits::arity>{});
        return FunctionResult::void_result();
      } else {
        auto result = call_with_args_impl<ArgsTuple>(
            f, args, std::make_index_sequence<FuncTraits::arity>{});
        return FunctionResult(result);
      }
    };
  }

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
    std::cout << "Registered schema: " << qualified_name << " -> " << schema
              << std::endl;
  }

  void register_implementation(const std::string& qualified_name,
                               DispatchKey key,
                               CppFunction&& func) {
    auto& op = get_or_create_operator(qualified_name);
    op.implementations[key] = std::move(func);
    std::cout << "Registered implementation: " << qualified_name << " for "
              << dispatch_key_to_string(key) << std::endl;
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
      std::cout << "Error: Operator " << qualified_name << " not found!"
                << std::endl;
      return false;
    }

    auto impl_it = op->implementations.find(key);
    if (impl_it != op->implementations.end()) {
      try {
        std::cout << "Executing " << qualified_name << " with "
                  << dispatch_key_to_string(key) << std::endl;
        auto result = impl_it->second.call();
        if (result.has_value()) {
          std::cout << "Operator executed successfully with return value"
                    << std::endl;
        } else {
          std::cout << "Operator executed successfully (void return)"
                    << std::endl;
        }
        return true;
      } catch (const std::exception& e) {
        std::cout << "Error executing operator: " << e.what() << std::endl;
        return false;
      }
    }

    // try fallback to CPU
    if (key != DispatchKey::CPU) {
      auto cpu_it = op->implementations.find(DispatchKey::CPU);
      if (cpu_it != op->implementations.end()) {
        std::cout << "Fallback to CPU for " << qualified_name << std::endl;
        try {
          auto result = cpu_it->second.call();
          if (result.has_value()) {
            std::cout << "Operator executed successfully with return value "
                         "(CPU fallback)"
                      << std::endl;
          } else {
            std::cout
                << "Operator executed successfully (void return, CPU fallback)"
                << std::endl;
          }
          return true;
        } catch (const std::exception& e) {
          std::cout << "Error executing operator (CPU fallback): " << e.what()
                    << std::endl;
          return false;
        }
      }
    }

    std::cout << "Error: No implementation found for " << qualified_name
              << " with " << dispatch_key_to_string(key) << std::endl;
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
        std::cout << "Executing " << qualified_name << " with "
                  << dispatch_key_to_string(key) << std::endl;
        auto result = impl_it->second.call(std::forward<Args>(args)...);
        if (result.has_value()) {
          std::cout << "Operator executed successfully with return value"
                    << std::endl;
        } else {
          std::cout << "Operator executed successfully (void return)"
                    << std::endl;
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
        std::cout << "Fallback to CPU for " << qualified_name << std::endl;
        try {
          auto result = cpu_it->second.call(std::forward<Args>(args)...);
          if (result.has_value()) {
            std::cout << "Operator executed successfully with return value "
                         "(CPU fallback)"
                      << std::endl;
          } else {
            std::cout
                << "Operator executed successfully (void return, CPU fallback)"
                << std::endl;
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
    std::cout << "Created Library: kind=" << kind_to_string(kind)
              << ", namespace=" << ns;
    if (dispatch_key) {
      std::cout << ", dispatch_key=" << dispatch_key_to_string(*dispatch_key);
    }
    std::cout << std::endl;
  }

  // 定义操作符 schema（用于 TORCH_LIBRARY 和 TORCH_LIBRARY_FRAGMENT）
  Library& def(const std::string& schema) & {
    if (kind_ == IMPL) {
      std::cout
          << "Warning: def() should not be called in TORCH_LIBRARY_IMPL block"
          << std::endl;
      return *this;
    }

    // 简单的 schema 解析：假设格式为 "op_name(args) -> return_type"
    auto op_name = extract_op_name(schema);
    auto qualified_name = ns_ + "::" + op_name;

    OperatorRegistry::instance().register_schema(qualified_name, schema);
    return *this;
  }

  // 定义操作符并立即提供实现
  template <typename Func>
  Library& def(const std::string& name_or_schema, Func&& f) & {
    auto op_name = extract_op_name(name_or_schema);
    auto qualified_name = ns_ + "::" + op_name;

    // 如果看起来像 schema，先注册 schema
    if (name_or_schema.find('(') != std::string::npos) {
      OperatorRegistry::instance().register_schema(qualified_name,
                                                   name_or_schema);
    }

    // 注册实现
    auto dispatch_key = dispatch_key_.value_or(DispatchKey::CPU);
    OperatorRegistry::instance().register_implementation(
        qualified_name, dispatch_key, CppFunction(std::forward<Func>(f)));

    return *this;
  }

  // 实现操作符（用于 TORCH_LIBRARY_IMPL）
  template <typename Func>
  Library& impl(const std::string& op_name, Func&& f) & {
    auto qualified_name = ns_ + "::" + op_name;
    auto dispatch_key = dispatch_key_.value_or(DispatchKey::CPU);

    OperatorRegistry::instance().register_implementation(
        qualified_name, dispatch_key, CppFunction(std::forward<Func>(f)));

    return *this;
  }

  // 打印当前库信息
  void print_info() const {
    std::cout << "Library Info: " << kind_to_string(kind_)
              << ", namespace=" << ns_;
    if (dispatch_key_) {
      std::cout << ", dispatch_key=" << dispatch_key_to_string(*dispatch_key_);
    }
    std::cout << std::endl;
  }

 private:
  Kind kind_;
  std::string ns_;
  std::optional<DispatchKey> dispatch_key_;
  const char* file_;
  uint32_t line_;

  std::string extract_op_name(const std::string& name_or_schema) const {
    // 简单的名称提取：如果包含'('，提取'('前的部分
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
    // 立即执行初始化（模拟 PyTorch 的静态初始化行为）
    Library lib(kind, ns, dispatch_key, file, line);
    fn(lib);
  }
};

}  // namespace detail

// 用于生成唯一标识符的宏
#define TORCH_CONCAT_IMPL(x, y) x##y
#define TORCH_CONCAT(x, y) TORCH_CONCAT_IMPL(x, y)
#define TORCH_UNIQUE_NAME(prefix) TORCH_CONCAT(prefix, __LINE__)

// TORCH_LIBRARY - 定义主库
#define TORCH_LIBRARY(ns, m)                                               \
  static void TORCH_UNIQUE_NAME(torch_library_init_)(torch::Library&);     \
  static const torch::detail::TorchLibraryInit TORCH_UNIQUE_NAME(          \
      torch_library_static_init_)(torch::Library::DEF,                     \
                                  &TORCH_UNIQUE_NAME(torch_library_init_), \
                                  #ns,                                     \
                                  std::nullopt,                            \
                                  __FILE__,                                \
                                  __LINE__);                               \
  void TORCH_UNIQUE_NAME(torch_library_init_)(torch::Library & m)  // NOLINT

// TORCH_LIBRARY_FRAGMENT - 定义库片段
#define TORCH_LIBRARY_FRAGMENT(ns, m)                                   \
  static void TORCH_UNIQUE_NAME(torch_library_fragment_init_)(          \
      torch::Library&);                                                 \
  static const torch::detail::TorchLibraryInit TORCH_UNIQUE_NAME(       \
      torch_library_fragment_static_init_)(                             \
      torch::Library::FRAGMENT,                                         \
      &TORCH_UNIQUE_NAME(torch_library_fragment_init_),                 \
      #ns,                                                              \
      std::nullopt,                                                     \
      __FILE__,                                                         \
      __LINE__);                                                        \
  void TORCH_UNIQUE_NAME(torch_library_fragment_init_)(torch::Library & \
                                                       m)  // NOLINT

// TORCH_LIBRARY_IMPL - 定义实现
#define TORCH_LIBRARY_IMPL(ns, key, m)                                      \
  static void TORCH_UNIQUE_NAME(torch_library_impl_init_)(torch::Library&); \
  static const torch::detail::TorchLibraryInit TORCH_UNIQUE_NAME(           \
      torch_library_impl_static_init_)(                                     \
      torch::Library::IMPL,                                                 \
      &TORCH_UNIQUE_NAME(torch_library_impl_init_),                         \
      #ns,                                                                  \
      torch::DispatchKey::key,                                              \
      __FILE__,                                                             \
      __LINE__);                                                            \
  void TORCH_UNIQUE_NAME(torch_library_impl_init_)(torch::Library &         \
                                                   m)  // NOLINT

}  // namespace torch
