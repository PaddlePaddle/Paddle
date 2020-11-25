// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

class InferNoNeedBufferVarsContext {
 public:
  explicit InferNoNeedBufferVarsContext(const framework::AttributeMap &attrs)
      : attrs_(attrs) {}
  virtual ~InferNoNeedBufferVarsContext() = default;

  virtual bool HasOutput(const std::string &slot) const = 0;

  const Attribute &GetAttr(const std::string &attr) const;

 private:
  const framework::AttributeMap &attrs_;
};

class StaticGraphInferNoNeedBufferVarsContext final
    : public InferNoNeedBufferVarsContext {
 public:
  StaticGraphInferNoNeedBufferVarsContext(const VariableNameMap &inputs,
                                          const VariableNameMap &outputs,
                                          const AttributeMap &attrs);

  bool HasOutput(const std::string &slot) const final;

 private:
  const VariableNameMap &inputs_;
  const VariableNameMap &outputs_;
};

class DyGraphInferNoNeedBufferVarsContext final
    : public InferNoNeedBufferVarsContext {
 public:
  DyGraphInferNoNeedBufferVarsContext(
      const imperative::NameVarMap<imperative::VariableWrapper> &inputs,
      const imperative::NameVarMap<imperative::VariableWrapper> &outputs,
      const AttributeMap &attrs);

  bool HasOutput(const std::string &slot) const final;

 private:
  const imperative::NameVarMap<imperative::VariableWrapper> &inputs_;
  const imperative::NameVarMap<imperative::VariableWrapper> &outputs_;
};

class NoNeedBufferVarsInference {
 public:
  virtual ~NoNeedBufferVarsInference() = default;
  virtual const std::unordered_set<std::string> &operator()(
      const InferNoNeedBufferVarsContext &ctx) const = 0;

 protected:
  static const std::unordered_set<std::string> &Empty() {
    static std::unordered_set<std::string> empty;
    return empty;
  }
};

#define DECLARE_NO_NEED_BUFFER_VARS_INFERER(class_type, ...)          \
  class class_type final                                              \
      : public ::paddle::framework::NoNeedBufferVarsInference {       \
   public:                                                            \
    using ::paddle::framework::NoNeedBufferVarsInference::            \
        NoNeedBufferVarsInference;                                    \
                                                                      \
    const std::unordered_set<std::string> &operator()(                \
        const ::paddle::framework::InferNoNeedBufferVarsContext &ctx) \
        const final {                                                 \
      static std::unordered_set<std::string> __ret__{__VA_ARGS__};    \
      return __ret__;                                                 \
    }                                                                 \
  }

class InferNoNeedBufferVarsFN {
 public:
  inline const std::unordered_set<std::string> &operator()(
      const VariableNameMap &inputs, const VariableNameMap &outputs,
      const AttributeMap &attrs) const {
    PADDLE_ENFORCE_NOT_NULL(
        inferer_,
        platform::errors::PreconditionNotMet(
            "The `inferer_` of InferNoNeedBufferVarsFN is not initialized."));
    StaticGraphInferNoNeedBufferVarsContext ctx(inputs, outputs, attrs);
    return (*inferer_)(ctx);
  }

  inline const std::unordered_set<std::string> &operator()(
      const imperative::NameVarMap<imperative::VariableWrapper> &inputs,
      const imperative::NameVarMap<imperative::VariableWrapper> &outputs,
      const AttributeMap &attrs) const {
    PADDLE_ENFORCE_NOT_NULL(
        inferer_,
        platform::errors::PreconditionNotMet(
            "The `inferer_` of InferNoNeedBufferVarsFN is not initialized."));
    DyGraphInferNoNeedBufferVarsContext ctx(inputs, outputs, attrs);
    return (*inferer_)(ctx);
  }

  inline explicit operator bool() const { return inferer_ != nullptr; }

  inline bool operator!() const { return inferer_ == nullptr; }

  inline void Reset(const std::shared_ptr<NoNeedBufferVarsInference> &inferer) {
    PADDLE_ENFORCE_NOT_NULL(
        inferer, platform::errors::InvalidArgument("The input inferer of "
                                                   "InferNoNeedBufferVarsFN::"
                                                   "Reset is nullptr."));
    PADDLE_ENFORCE_EQ(
        inferer_, nullptr,
        platform::errors::AlreadyExists(
            "The `inferer_` of InferNoNeedBufferVarsFN has been initialized."));
    inferer_ = inferer;
  }

  inline bool operator==(std::nullptr_t) const { return inferer_ == nullptr; }

  inline bool operator!=(std::nullptr_t) const { return inferer_ != nullptr; }

 private:
  std::shared_ptr<NoNeedBufferVarsInference> inferer_;
};

static inline bool operator==(std::nullptr_t,
                              const InferNoNeedBufferVarsFN &other) {
  return other == nullptr;
}

static inline bool operator!=(std::nullptr_t,
                              const InferNoNeedBufferVarsFN &other) {
  return other != nullptr;
}

}  // namespace framework
}  // namespace paddle
