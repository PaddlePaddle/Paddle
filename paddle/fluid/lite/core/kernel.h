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

  // Get input declaration Type.
  const Type* GetInputDeclType(const std::string& arg_name);

  // Get output declaration Type.
  const Type* GetOutputDeclType(const std::string& arg_name);

  void set_alias(const std::string& x) { alias_ = x; }
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
  // Generate the key of the parameter type.
  std::string GenParamTypeKey() const;

  std::string SerializedKernelType() const {
    return SerializeKernelType(op_type(), alias(), place());
  }

  static std::string SerializeKernelType(const std::string& op_type,
                                         const std::string& alias,
                                         const Place& place) {
    std::stringstream ss;
    ss << op_type << "/";
    ss << alias << "/";
    // We serialize the place value not the string representation here for
    // easier deserialization.
    ss << static_cast<int>(place.target) << "/";
    ss << static_cast<int>(place.precision) << "/";
    ss << static_cast<int>(place.layout);
    return ss.str();
  }

  static void ParseKernelType(const std::string& kernel_type,
                              std::string* op_type, std::string* alias,
                              Place* place) {
    std::stringstream ss(kernel_type);
    std::getline(ss, *op_type, '/');
    std::getline(ss, *alias, '/');
    std::string target, precision, layout;
    std::getline(ss, target, '/');
    std::getline(ss, precision, '/');
    std::getline(ss, layout, '/');

    place->target = static_cast<TargetType>(std::stoi(target));
    place->precision = static_cast<PrecisionType>(std::stoi(precision));
    place->layout = static_cast<DataLayoutType>(std::stoi(layout));
  }

  virtual ~KernelBase() = default;
  void Torch() {}

 protected:
  std::unique_ptr<KernelContext> context_;
  mutable operators::param_t param_;
  // The corresponding op type.
  std::string op_type_{};
  // The extra identity to help defficiate a specific kernel, op_type_ + alias_
  // is the unique ID for the kernel.
  std::string alias_{};
};

// Light-weight kernel implementation.
// The OpKernel is designed to implement the specific algorithm on a target
// device.
// TODO(Superjomn) Consider to add a Platform type to differentiate CUDNN,
// MKLDNN, plain CUDA C implementations.
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
  std::string name() const override;

  void Touch() {}

  OpKernel() = default;
  virtual ~OpKernel() = default;

 protected:
  std::unique_ptr<KernelContext> ctx_;
};

template <TargetType Target, PrecisionType Precision, DataLayoutType DataLayout>
std::string OpKernel<Target, Precision, DataLayout>::name() const {
  return op_type() + ":" + TargetToStr(Target) + "/" +
         PrecisionToStr(Precision) + "/" + DataLayoutToStr(DataLayout);
}

}  // namespace lite
}  // namespace paddle
