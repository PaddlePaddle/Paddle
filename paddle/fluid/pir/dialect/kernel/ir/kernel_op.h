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

#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/op_base.h"

namespace paddle {
namespace dialect {
class PhiKernelOp : public pir::Op<PhiKernelOp> {
 public:
  using Op::Op;
  static const char *name() { return "pd_kernel.phi_kernel"; }
  static constexpr uint32_t attributes_num = 3;
  static const char *attributes_name[attributes_num];
  std::string op_name();
  std::string kernel_name();
  phi::KernelKey kernel_key();
  void Verify();
};

class LegacyKernelOp : public pir::Op<LegacyKernelOp> {
 public:
  using Op::Op;
  static const char *name() { return "pd_kernel.legacy_kernel"; }
  static constexpr uint32_t attributes_num = 3;
  static const char *attributes_name[attributes_num];
  std::string op_name();
  std::string kernel_name();
  phi::KernelKey kernel_key();
  void Verify();
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PhiKernelOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::LegacyKernelOp)
