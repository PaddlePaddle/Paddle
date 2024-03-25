// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/pir/include/core/op_base.h"

namespace cinn {

namespace dialect {

class JitKernelOp : public ::pir::Op<JitKernelOp> {
 public:
  using Op::Op;
  static const char* name() { return "cinn_runtime.jit_kernel"; }
  // TODO(Aurelius84): Think deeply what should contains
  static constexpr uint32_t attributes_num = 1;
  static constexpr char* kAttrName = "kernel_info";
  static const char* attributes_name[attributes_num];

  static void Build(::pir::Builder& builder,             // NOLINT
                    ::pir::OperationArgument& argument,  // NOLINT
                    const std::vector<::pir::Value>& x,
                    const ::pir::AttributeMap& attributes,
                    const std::vector<::pir::Type>& out_types);

  const hlir::framework::pir::CINNKernelInfo& cinn_kernel_info();

  void VerifySig();
};

}  // namespace dialect
}  // namespace cinn

IR_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::JitKernelOp)
