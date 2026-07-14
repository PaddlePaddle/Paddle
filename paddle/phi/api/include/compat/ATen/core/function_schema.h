// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once
#include <ATen/core/alias_info.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <string>
#include <vector>
#include "paddle/common/macros.h"  // For macro PADDLE_API

namespace c10 {

struct Argument;
struct FunctionSchema;
enum class SchemaArgType;
struct SchemaArgument;

enum class SchemaArgType {
  input,
  output,
};

struct SchemaArgument {
  SchemaArgType type;
  size_t index;
};

struct PADDLE_API Argument {
  Argument(std::string name = "",
           const TypePtr& type = nullptr,
           std::optional<int32_t> N = std::nullopt,
           std::optional<torch::IValue> default_value = std::nullopt,
           bool kwarg_only = false,
           std::optional<c10::AliasInfo> alias_info = std::nullopt)
      : Argument(std::move(name),
                 type,
                 type,
                 N,
                 std::move(default_value),
                 kwarg_only,
                 std::move(alias_info)) {}

  Argument(std::string name,
           TypePtr fake_type,
           TypePtr real_type,
           std::optional<int32_t> N = std::nullopt,
           std::optional<torch::IValue> default_value = std::nullopt,
           bool kwarg_only = false,
           std::optional<c10::AliasInfo> alias_info = std::nullopt)
      : name_(std::move(name)),
        type_(fake_type ? std::move(fake_type) : TensorType::get()),
        real_type_(real_type ? std::move(real_type) : type_),
        N_(N),
        default_value_(std::move(default_value)),
        alias_info_(alias_info ? std::make_unique<c10::AliasInfo>(
                                     std::move(*alias_info))
                               : nullptr),
        kwarg_only_(kwarg_only) {
    // this is an softly-enforced invariant for out arguments.
    bool is_alias = alias_info_ != nullptr && alias_info_->isWrite();
    is_out_ = kwarg_only_ && is_alias;
  }

  Argument(Argument&& rhs) noexcept = default;

  Argument(const Argument& rhs)
      : name_(rhs.name_),
        type_(rhs.type_),
        real_type_(rhs.real_type_),
        N_(rhs.N_),
        default_value_(rhs.default_value_),
        alias_info_(rhs.alias_info_
                        ? std::make_unique<c10::AliasInfo>(*rhs.alias_info_)
                        : nullptr),
        kwarg_only_(rhs.kwarg_only_),
        is_out_(rhs.is_out_) {}

  Argument& operator=(Argument&& rhs) = default;

  Argument& operator=(const Argument& rhs) {
    if (this != &rhs) {
      name_ = rhs.name_;
      type_ = rhs.type_;
      real_type_ = rhs.real_type_;
      N_ = rhs.N_;
      default_value_ = rhs.default_value_;
      alias_info_ = rhs.alias_info_
                        ? std::make_unique<c10::AliasInfo>(*rhs.alias_info_)
                        : nullptr;
      kwarg_only_ = rhs.kwarg_only_;
      is_out_ = rhs.is_out_;
    }
    return *this;
  }
  ~Argument() = default;

  const std::string& name() const { return name_; }
  const TypePtr& type() const { return type_; }
  // if type() is non-null, this is guaranteed to be non-null (if no real
  // type was provided, this takes on type()'s value)
  const TypePtr& real_type() const { return real_type_; }
  const std::optional<int32_t>& N() const { return N_; }
  const std::optional<torch::IValue>& default_value() const {
    return default_value_;
  }
  bool kwarg_only() const { return kwarg_only_; }

  bool is_out() const { return is_out_; }

  [[nodiscard]] const c10::AliasInfo* alias_info() const {
    return alias_info_.get();
  }

  bool is_inferred_type() const {
    bool is_inferred_type = false;
    TORCH_INTERNAL_ASSERT(type_);
    if (auto pt = type_->cast<TensorType>()) {
      if (pt->isInferredType()) {
        is_inferred_type = true;
      }
    }
    return is_inferred_type;
  }

  std::string formatTypeMismatchMsg(const std::string& actual_type) const {
    std::string inferred_type_hint;
    if (is_inferred_type()) {
      inferred_type_hint = "Inferred '";
      inferred_type_hint += name();
      inferred_type_hint += "' to be of type 'Tensor' ";
      inferred_type_hint +=
          "because it was not annotated with an explicit type.\n";
    }
    std::string message;
    message += "Expected a value of type '";
    message += type()->repr_str();
    message += "' for argument '";
    message += name();
    message += "' but instead found type '";
    message += actual_type;
    message += "'.\n";
    message += inferred_type_hint;
    return message;
  }

  Argument cloneWithType(const TypePtr& new_type) const {
    return Argument(name_,
                    new_type,
                    N_,
                    default_value_,
                    kwarg_only_,
                    alias_info_ ? std::optional<c10::AliasInfo>(*alias_info_)
                                : std::nullopt);
  }

  friend PADDLE_API std::ostream& operator<<(std::ostream& out,
                                             const Argument& arg);

 private:
  std::string name_;
  TypePtr type_;
  TypePtr real_type_;  // this is ScalarType, not int, e.g.
  // for list types, an optional statically known length for the list
  // e.g. for int[3]: type = ListType::ofInts(), N = 3
  // If present, this will allow scalars to be broadcast to this length to
  // become a list.
  std::optional<int32_t> N_;

  std::optional<torch::IValue> default_value_;
  // c10::AliasInfo is huge, so let's only allocate memory for it if
  // necessary (which it isn't during schema parsing on startup, to
  // give a pertinent example).
  std::unique_ptr<c10::AliasInfo> alias_info_;
  // is this only specifiable as a keyword argument?
  bool kwarg_only_;
  // marks if the argument is out variant of the schema
  bool is_out_;
};

struct PADDLE_API FunctionSchema {
  FunctionSchema(std::vector<Argument> arguments,
                 std::vector<Argument> returns,
                 bool is_vararg = false,
                 bool is_varret = false)
      : arguments_(std::move(arguments)),
        returns_(std::move(returns)),
        is_vararg_(is_vararg),
        is_varret_(is_varret) {
    checkSchema();
  }

  const std::vector<Argument>& arguments() const { return arguments_; }

  void checkSchema() const {
    bool seen_default_arg = false;
    for (const auto& arg : arguments()) {
      if (arg.default_value()) {
        seen_default_arg = true;
      } else {
        TORCH_INTERNAL_ASSERT(!seen_default_arg || arg.kwarg_only(),
                              "Non-default positional argument follows default "
                              "argument. Parameter ",
                              arg.name(),
                              " in ",
                              *this);
      }
    }
  }

  const std::vector<Argument>& returns() const { return returns_; }

  bool is_vararg() const { return is_vararg_; }

  bool is_varret() const { return is_varret_; }

  std::optional<int> argumentIndexWithName(const std::string& name) const;
  const std::vector<Argument>& getCorrectList(
      const SchemaArgument& argument) const;
  bool is_aliasing(const SchemaArgument& argument) const;
  bool is_mutable(const SchemaArgument& argument) const;
  bool is_mutable(const std::string& name) const;
  bool may_alias(const SchemaArgument& lhs, const SchemaArgument& rhs) const;
  bool may_contain_alias(const SchemaArgument& lhs,
                         const SchemaArgument& rhs,
                         bool bidirectional = true) const;

  friend PADDLE_API std::ostream& operator<<(std::ostream& out,
                                             const FunctionSchema& schema);

 private:
  std::vector<Argument> arguments_;
  std::vector<Argument> returns_;
  // if true then this schema takes an arbitrary number of additional arguments
  // after the argument specified in arguments
  // currently this is used primarily to represent 'primitive' operators whose
  // arguments are not checked by schema
  bool is_vararg_;
  bool is_varret_;
};

PADDLE_API std::ostream& operator<<(std::ostream& out, const Argument& arg);
PADDLE_API std::ostream& operator<<(std::ostream& out,
                                    const FunctionSchema& schema);

}  // namespace c10
