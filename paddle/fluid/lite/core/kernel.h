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

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/core/type_system.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

// An base with virtual functions to unify all the kernel implementation on
// different targets.
class KernelBase {
 public:
  // type_infer_handler is used to inference a output type by considering the
  // input types in the type system.
  using type_infer_handler_t = std::function<const Type*(
      const std::map<std::string, const Type*>& input_types,
      const std::string& out_arg)>;

  virtual void Run() = 0;

  void SetContext(std::unique_ptr<KernelContext>&& ctx) {
    context_ = std::move(ctx);
  }
  template <typename T>
  void SetParam(T param) {
    param_.set<T>(param);
  }
  template <typename P>
  P& Param() const {
    return param_.get<P>();
  }

  // This is used in the kernels that takes 'kAny' places and inference the
  // output place. For `ScaleCompute` and `IoCopyCompute`, their input types are
  // declared as 'kAny' in some Place field, and the output is also `kAny`, but
  // when in real execution, when takes some non-kAny type as input, the
  // output's kAny-fields can be determained. For example, when the
  // `ScaleCompute` takes `TensorFp32NCHWTy` as input, its output should be also
  // `TensorFp32NCHWTy`. This type inference rule is different for each kernel,
  // so we make it a virtual method.
  // One can custom this handler to make a specific type inference rule for a
  // kernel, or leave the default to force the kernel use the system's
  // type-inference rules.
  virtual std::unique_ptr<type_infer_handler_t> GetTypeInferHandler() {
    return nullptr;
  }

  void set_op_type(const std::string& type) { op_type_ = type; }
  const std::string& op_type() const { return op_type_; }

  void Torch() {}

  // Get input declaration type.
  const Type* GetInputDeclType(const std::string& arg_name) {
    CHECK(!op_type_.empty()) << "op_type should be set first";
    const auto* type = ParamTypeRegistry::Global().RetrieveInArgument(
        place(), GenParamTypeKey(), arg_name);
    CHECK(type) << "no type registered for kernel [" << op_type_
                << "] input argument [" << arg_name << "]"
                << " with key " << GenParamTypeKey();
    return type->type;
  }

  // Get output declaration type.
  const Type* GetOutputDeclType(const std::string& arg_name) {
    CHECK(!op_type_.empty()) << "op_type should be set first";
    const auto* type = ParamTypeRegistry::Global().RetrieveOutArgument(
        place(), GenParamTypeKey(), arg_name);
    CHECK(type) << "no type registered for kernel [" << op_type_
                << "] output argument [" << arg_name << "]";
    return type->type;
  }

  void set_alias(const std::string& x) {
    alias_ = x;
    LOG(INFO) << "kernel " << op_type() << " setting alias " << alias();
  }
  const std::string& alias() const { return alias_; }

  virtual Place place() const = 0;
  virtual TargetType target() const = 0;
  virtual PrecisionType precision() const = 0;
  virtual DataLayoutType layout() const = 0;
  const KernelContext* context() const { return context_.get(); }
  virtual std::string name() const = 0;

  // Short human-readable document.
  std::string summary() const;
  // Long human-readable document.
  virtual std::string doc() const { return ""; }

  std::string GenParamTypeKey() const {
    std::stringstream ss;
    LOG(INFO) << "alias : " << alias_;
    ss << op_type() << "/" << alias_;
    return ss.str();
  }

  virtual ~KernelBase() = default;

 protected:
  std::unique_ptr<KernelContext> context_;
  mutable operators::param_t param_;
  // The corresponding op type.
  std::string op_type_{};
  std::string alias_{};
};

// Light-weight kernel implementation.
// The OpKernel is designed to implement the specific algorithm on a target
// device.
template <TargetType Target, PrecisionType Precision,
          DataLayoutType DataLayout = DataLayoutType::kNCHW>
class OpKernel : public KernelBase {
 public:
  // Set runtime context.
  void SetContext(std::unique_ptr<KernelContext>&& ctx) { ctx_ = ctx; }

  // Run the kernel.
  virtual void Run() { CHECK(false) << "Not Implemented"; }

  TargetType target() const override { return Target; }
  PrecisionType precision() const override { return Precision; }
  DataLayoutType layout() const override { return DataLayout; }
  Place place() const override { return Place{Target, Precision, DataLayout}; }
  std::string name() const override {
    return op_type() + ":" + TargetToStr(Target) + "/" +
           PrecisionToStr(Precision) + "/" + DataLayoutToStr(DataLayout);
  }

  void Touch() {}

  OpKernel() = default;
  virtual ~OpKernel() = default;

 protected:
  std::unique_ptr<KernelContext> ctx_;
};

}  // namespace lite
}  // namespace paddle
