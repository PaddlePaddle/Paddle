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
#include <string>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
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

  template <TargetType Target>
  void SetContext(std::unique_ptr<Context<Target>>&& ctx) {
    context_.set<std::unique_ptr<Context<Target>>>(std::move(ctx));
  }

  template <typename T>
  void SetParam(T param) {
    param_.set<T>(param);
  }

  template <typename Param>
  Param& param() const {
    return param_.get<Param>();
  }

  virtual TargetType target() const = 0;
  virtual PrecisionType precision() const = 0;

  virtual ~KernelBase() = default;

 protected:
  core::any_context_t context_;
  mutable operators::param_t param_;
};

// Light-weight kernel implementation.
// The OpKernel is designed to implement the specific algorithm on a target
// device.
template <TargetType Target, PrecisionType Precision>
class OpKernel : public KernelBase {
 public:
  virtual void Run() { CHECK(false) << "Not Implemented"; }

  OpKernel() = default;

  virtual ~OpKernel() = default;
};

}  // namespace lite
}  // namespace paddle
