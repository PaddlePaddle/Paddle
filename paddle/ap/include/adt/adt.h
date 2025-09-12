// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <functional>
#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "glog/logging.h"
#include "paddle/common/overloaded.h"

namespace ap::adt {

template <typename T>
struct Rc {
 public:
  Rc() : data_(std::make_shared<T>()) {}
  explicit Rc(const std::shared_ptr<T>& data) : data_(data) {}
  Rc(const Rc&) = default;
  Rc(Rc&&) = default;
  Rc& operator=(const Rc&) = default;
  Rc& operator=(Rc&&) = default;

  template <typename Arg,
            std::enable_if_t<
                !std::is_same_v<std::decay_t<Arg>, Rc> &&
                    !std::is_same_v<std::decay_t<Arg>, std::shared_ptr<T>>,
                bool> = true>
  explicit Rc(Arg&& arg) : data_(new T{std::forward<Arg>(arg)}) {}

  template <typename Arg0, typename Arg1, typename... Args>
  explicit Rc(Arg0&& arg0, Arg1&& arg1, Args&&... args)
      : data_(new T{std::forward<Arg0>(arg0),
                    std::forward<Arg1>(arg1),
                    std::forward<Args>(args)...}) {}

  T* operator->() { return data_.get(); }
  const T* operator->() const { return data_.get(); }

  T& operator*() { return *data_; }
  const T& operator*() const { return *data_; }

  bool operator==(const Rc& other) const {
    if (other.data_.get() == this->data_.get()) {
      return true;
    }
    return *other.data_ == *this->data_;
  }

  const std::shared_ptr<T>& shared_ptr() const { return data_; }

  const void* __adt_rc_shared_ptr_raw_ptr() const { return data_.get(); }

 private:
  std::shared_ptr<T> data_;
};

#define ADT_DEFINE_RC(class_name, ...)                    \
  struct class_name : public ::ap::adt::Rc<__VA_ARGS__> { \
    using ::ap::adt::Rc<__VA_ARGS__>::Rc;                 \
  };

#define ADT_DEFINE_VARIANT_METHODS(...)                                \
  ADT_DEFINE_VARIANT_METHODS_WITHOUT_TRYGET(__VA_ARGS__)               \
  template <typename __AlternativeT, int start_idx = 0>                \
  static constexpr bool IsMyAlternative() {                            \
    if constexpr (start_idx >= std::variant_size_v<__VA_ARGS__>) {     \
      return false;                                                    \
    } else {                                                           \
      using AlternativeT =                                             \
          typename std::variant_alternative_t<start_idx, __VA_ARGS__>; \
      if constexpr (std::is_same_v<AlternativeT, __AlternativeT>) {    \
        return true;                                                   \
      } else {                                                         \
        return IsMyAlternative<__AlternativeT, start_idx + 1>();       \
      }                                                                \
    }                                                                  \
  }                                                                    \
  template <typename __ADT_T>                                          \
  ::ap::adt::Result<__ADT_T> TryGet() const {                          \
    ADT_CHECK(this->template Has<__ADT_T>());                          \
    return this->template Get<__ADT_T>();                              \
  }

#define ADT_DEFINE_VARIANT_METHODS_WITHOUT_TRYGET(...)                 \
  DEFINE_MATCH_METHOD();                                               \
  const __VA_ARGS__& variant() const {                                 \
    return reinterpret_cast<const __VA_ARGS__&>(*this);                \
  }                                                                    \
  template <typename __ADT_T>                                          \
  bool Has() const {                                                   \
    return std::holds_alternative<__ADT_T>(variant());                 \
  }                                                                    \
  template <typename __ADT_T>                                          \
  const __ADT_T& Get() const {                                         \
    return std::get<__ADT_T>(variant());                               \
  }                                                                    \
  bool operator!=(const __VA_ARGS__& other) const {                    \
    return !(*this == other);                                          \
  }                                                                    \
  bool operator==(const __VA_ARGS__& other) const {                    \
    return std::visit(                                                 \
        [](const auto& lhs, const auto& rhs) {                         \
          if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,    \
                                       std::decay_t<decltype(rhs)>>) { \
            return lhs == rhs;                                         \
          } else {                                                     \
            return false;                                              \
          }                                                            \
        },                                                             \
        this->variant(),                                               \
        other);                                                        \
  }

template <typename T>
class List final {
 public:
  List(const List&) = default;
  List(List&&) = default;
  List& operator=(const List&) = default;
  List& operator=(List&&) = default;

  using value_type = T;

  explicit List() : vector_(std::make_shared<std::vector<T>>()) {}

  template <
      typename Arg,
      std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, List>, bool> = true>
  explicit List(Arg&& arg)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg>(arg)})) {}

  template <typename Arg0, typename Arg1, typename... Args>
  List(Arg0&& arg0, Arg1&& arg1, Args&&... args)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg0>(arg0),
                           std::forward<Arg1>(arg1),
                           std::forward<Args>(args)...})) {}

  bool operator==(const List& other) const {
    if (&vector() == &other.vector()) {
      return true;
    }
    return vector() == other.vector();
  }

  bool operator!=(const List& other) const { return !(*this == other); }

  std::vector<T>& operator*() const { return *vector_; }
  std::vector<T>* operator->() const { return vector_.get(); }

  const std::vector<T>& vector() const { return *vector_; }

  const auto& Get(std::size_t idx) const { return vector_->at(idx); }

 private:
  std::shared_ptr<std::vector<T>> vector_;
};

#define ADT_DEFINE_TAG(TagName)                                             \
  template <typename T>                                                     \
  class TagName {                                                           \
   public:                                                                  \
    TagName() = default;                                                    \
    TagName(const TagName&) = default;                                      \
    TagName(TagName&&) = default;                                           \
    TagName& operator=(const TagName&) = default;                           \
    TagName& operator=(TagName&&) = default;                                \
                                                                            \
    bool operator==(const TagName& other) const {                           \
      return value_ == other.value();                                       \
    }                                                                       \
                                                                            \
    bool operator!=(const TagName& other) const {                           \
      return value_ != other.value();                                       \
    }                                                                       \
                                                                            \
    template <typename Arg,                                                 \
              std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, TagName>, \
                               bool> = true>                                \
    explicit TagName(Arg&& value) : value_(value) {}                        \
                                                                            \
    const T& value() const { return value_; }                               \
                                                                            \
   private:                                                                 \
    T value_;                                                               \
  };

// Undefined = {}
struct Undefined final : public std::monostate {
  using std::monostate::monostate;
};

// Ok = {}
struct Ok final : public std::monostate {
  using std::monostate::monostate;
};

inline std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
}

struct Nothing : public std::monostate {
  using std::monostate::monostate;
};

struct IdentityFunc : public std::monostate {
  using std::monostate::monostate;
};

template <typename T0, typename T1>
using EitherImpl = std::variant<T0, T1>;

template <typename T0, typename T1>
struct Either : public EitherImpl<T0, T1> {
  using EitherImpl<T0, T1>::EitherImpl;
  ADT_DEFINE_VARIANT_METHODS_WITHOUT_TRYGET(EitherImpl<T0, T1>);
};

template <typename T>
struct Maybe : public Either<T, Nothing> {
  using Either<T, Nothing>::Either;
};

namespace source_code {

struct CodeLocation {
  std::string file_name;
  int line_no;
  std::string func_name;
  std::string code;

  bool operator==(const CodeLocation& other) const {
    return this->file_name == other.file_name &&
           this->line_no == other.line_no &&
           this->func_name == other.func_name && this->code == other.code;
  }
};

using CallStack = std::list<const CodeLocation*>;

}  // namespace source_code

namespace errors {

struct RuntimeError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const RuntimeError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "RuntimeError"; }
};

struct InvalidArgumentError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const InvalidArgumentError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "InvalidArgumentError"; }
};

struct AttributeError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const AttributeError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "AttributeError"; }
};

struct NameError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const NameError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "NameError"; }
};

struct ValueError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const ValueError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "ValueError"; }
};

struct ZeroDivisionError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const ZeroDivisionError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "ZeroDivisionError"; }
};

struct TypeError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const TypeError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "TypeError"; }
};

struct IndexError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const IndexError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "IndexError"; }
};

struct KeyError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const KeyError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "KeyError"; }
};

struct MismatchError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const MismatchError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "MismatchError"; }
};

struct NotImplementedError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const NotImplementedError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "NotImplementedError"; }
};

struct SyntaxError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const SyntaxError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "SyntaxError"; }
};

struct ModuleNotFoundError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const ModuleNotFoundError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "ModuleNotFoundError"; }
};

struct AssertionError {
  std::string msg;
  source_code::CallStack call_stack{};

  bool operator==(const AssertionError& other) const {
    return this->msg == other.msg && this->call_stack == other.call_stack;
  }

  const char* class_name() const { return "AssertionError"; }
};

using ErrorBase = std::variant<RuntimeError,
                               InvalidArgumentError,
                               AttributeError,
                               NameError,
                               ValueError,
                               ZeroDivisionError,
                               TypeError,
                               IndexError,
                               KeyError,
                               MismatchError,
                               NotImplementedError,
                               SyntaxError,
                               ModuleNotFoundError,
                               AssertionError>;

struct [[nodiscard]] Error : public ErrorBase {
  using ErrorBase::ErrorBase;
  ADT_DEFINE_VARIANT_METHODS_WITHOUT_TRYGET(ErrorBase);

  const char* class_name() const {
    return Match([](const auto& impl) { return impl.class_name(); });
  }

  const std::string& msg() const {
    return Match(
        [](const auto& impl) -> const std::string& { return impl.msg; });
  }

  const source_code::CallStack& call_stack() const {
    return Match([](const auto& impl) -> const source_code::CallStack& {
      return impl.call_stack;
    });
  }

  std::string CallStackToString() const {
    std::ostringstream ss;
    for (const auto* code_location : call_stack()) {
      ss << "  File \"" << code_location->file_name << "\", line "
         << code_location->line_no << ", in " << code_location->func_name
         << "\n    " << code_location->code << "\n";
    }
    return ss.str();
  }

  Error operator<<(Error&& replacement) const {
    if (this->call_stack().size() > 0) {
      replacement.mut_call_stack()->push_front(*this->call_stack().begin());
    }
    return std::move(replacement);
  }

  Error operator<<(const Error& replacement) const {
    if (this->call_stack().size() > 0) {
      replacement.mut_call_stack()->push_front(*this->call_stack().begin());
    }
    return replacement;
  }

  Error operator<<(
      const std::function<Error(const Error&)>& GetReplacement) const {
    const auto& replacement = GetReplacement(*this);
    return (*this) << replacement;
  }

  Error operator<<(const source_code::CodeLocation* code_location) const {
    mut_call_stack()->push_front(code_location);
    return *this;
  }

 private:
  source_code::CallStack* mut_call_stack() const {
    return const_cast<source_code::CallStack*>(&call_stack());
  }
};

}  // namespace errors

template <typename T>
struct [[nodiscard]] Result : public Either<T, errors::Error> {
  using Either<T, errors::Error>::Either;

  bool HasError() const { return this->template Has<errors::Error>(); }

  bool HasOkValue() const { return !HasError(); }

  const errors::Error& GetError() const {
    return this->template Get<errors::Error>();
  }

  const T& GetOkValue() const { return this->template Get<T>(); }
};

struct Break : public std::monostate {
  using std::monostate::monostate;
};

struct Continue : public std::monostate {
  using std::monostate::monostate;
};

using LoopCtrlImpl = std::variant<Break, Continue>;

struct LoopCtrl : public LoopCtrlImpl {
  using LoopCtrlImpl::LoopCtrlImpl;

  ADT_DEFINE_VARIANT_METHODS(LoopCtrlImpl);
};

template <typename T>
adt::Result<std::shared_ptr<T>> WeakPtrLock(const std::weak_ptr<T>& weak_ptr) {
  const auto& ptr = weak_ptr.lock();
  if (!ptr) {
    return errors::RuntimeError{"weak_ptr.lock() failed."};
  }
  return ptr;
}

#define ADT_CURRENT_CODE_LOCATION(filename, line_no, func_name, code) \
  ([] {                                                               \
    static const ::ap::adt::source_code::CodeLocation loc{            \
        filename, line_no, func_name, code};                          \
    return &loc;                                                      \
  }())

// clang-format off
#define ADT_CHECK(...)                                             /* NOLINT */ \
  if (!(__VA_ARGS__))                                              /* NOLINT */ \
    return ::ap::adt::errors::Error{::ap::adt::errors::ValueError{ /* NOLINT */ \
        "Check '" #__VA_ARGS__ "' failed."                         /* NOLINT */ \
    }} << ADT_CURRENT_CODE_LOCATION(                               /* NOLINT */ \
      __FILE__, __LINE__, __FUNCTION__, #__VA_ARGS__               /* NOLINT */ \
    )
// clang-format on

#define ADT_RETURN_IF_ERR(...)                                       \
  if (const auto& __result##__LINE__ = __VA_ARGS__;                  \
      __result##__LINE__.HasError())                                 \
  return __result##__LINE__.GetError() << ADT_CURRENT_CODE_LOCATION( \
             __FILE__, __LINE__, __FUNCTION__, #__VA_ARGS__)

#define ADT_LET_CONST_REF(var, ...)                                         \
  const auto& __result_##var = __VA_ARGS__;                                 \
  const auto* __ptr_##var =                                                 \
      (__result_##var.HasError() ? nullptr : &__result_##var.GetOkValue()); \
  const auto& var = *__ptr_##var;                                           \
  if (__result_##var.HasError())                                            \
  return __result_##var.GetError() << ADT_CURRENT_CODE_LOCATION(            \
             __FILE__, __LINE__, __FUNCTION__, #__VA_ARGS__)

}  // namespace ap::adt
