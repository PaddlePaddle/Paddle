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

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/cinn_value.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace lang {
using cinn::common::CINNValue;

/**
 * A single argument value to Function.
 */
using ArgValue = CINNValue;

using RetValue = CINNValue;

/**
 * Arguments of the PackedFunc.
 */
class Args {
 public:
  Args() = default;
  Args(cinn_value_t* values, int* type_codes, int len);

  //! Append a \p value of type code \p type_code.
  void Append(const ArgValue& arg) { values_.push_back(arg); }

  //! Count of the arguments.
  size_t size() const { return values_.size(); }

  //! if the arguments is empty
  bool empty() const { return values_.empty(); }

  //! Get i-th element.
  ArgValue& operator[](int i) { return values_[i]; }
  const ArgValue& operator[](int i) const { return values_[i]; }

  cinn::common::CINNValuePack ToValuePack() const {
    return cinn::common::CINNValuePack(values_);
  }

 private:
  std::vector<ArgValue> values_;
};

namespace detail {

template <bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template <typename T, typename... Args>
  static void Run(const F& f, T&& value, Args&&... args) {
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, I + 1, F>::Run(
        f, std::forward<Args>(args)...);
  }
};

template <std::size_t I, typename F>
struct for_each_dispatcher<true, I, F> {
  static void Run(const F& f) {}
};

template <typename F, typename... Args>
inline void for_each(const F& f, Args&&... args) {
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>::Run(
      f, std::forward<Args>(args)...);
}

struct FuncArgsSetter {
  FuncArgsSetter(Args* args) : args_(args) {}  // NOLINT

  template <typename T>
  void operator()(size_t I, T v) const {
    args_->Append(ArgValue(v));
  }

 private:
  mutable Args* args_{};
};

}  // namespace detail

/**
 * A function defininer with the arguments packed, all the PackedFuncs have the
 * same signature.
 */
class PackedFunc {
 public:
  using body_t = std::function<void(Args args, RetValue*)>;

  PackedFunc() = default;
  explicit PackedFunc(const std::string& name) : name_(name) {}
  explicit PackedFunc(body_t body) : body_(body) {}

  template <typename... Args_>
  inline RetValue operator()(Args_&&... args) const {
    Args _args;
    detail::FuncArgsSetter setter(&_args);
    detail::for_each(setter, std::forward<Args_>(args)...);

    RetValue ret_value;
    body_(_args, &ret_value);
    return ret_value;
  }

  inline body_t body() const { return body_; }

 private:
  std::string name_;
  body_t body_;
};

}  // namespace lang
}  // namespace cinn
