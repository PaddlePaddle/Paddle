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

  void set_op_type(const std::string& type) { op_type_ = type; }
  const std::string& op_type() const { return op_type_; }

  void Torch() {}

  virtual Place place() const = 0;
  virtual TargetType target() const = 0;
  virtual PrecisionType precision() const = 0;
  virtual DataLayoutType layout() const = 0;
  const KernelContext* context() const { return context_.get(); }

  virtual std::string name() const = 0;

  virtual ~KernelBase() = default;

 protected:
  std::unique_ptr<KernelContext> context_;
  mutable operators::param_t param_;
  // The corresponding op type.
  std::string op_type_;
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
