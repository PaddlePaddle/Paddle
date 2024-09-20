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

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/op_base.h"

using KernelKeyTuple = std::tuple<phi::DataType, phi::Backend>;

namespace paddle {
namespace dialect {
class ParseKernelKeyInterface
    : public pir::OpInterfaceBase<ParseKernelKeyInterface> {
 public:
  struct Concept {
    explicit Concept(KernelKeyTuple (*parse_kernel_key)(pir::Operation *op))
        : parse_kernel_key_(parse_kernel_key) {}
    KernelKeyTuple (*parse_kernel_key_)(pir::Operation *op);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static KernelKeyTuple ParseKernelKey(pir::Operation *op) {
      return ConcreteOp::ParseKernelKey(op);
    }

    Model() : Concept(ParseKernelKey) {}
  };

  /// Constructor
  ParseKernelKeyInterface(const pir::Operation *op, Concept *impl)
      : pir::OpInterfaceBase<ParseKernelKeyInterface>(op), impl_(impl) {}

  KernelKeyTuple ParseKernelKey(pir::Operation *op) {
    return impl_->parse_kernel_key_(op);
  }

 private:
  Concept *impl_;
};

// Register the ParseKernelKeyInterface for unique op.
KernelKeyTuple UniqueOpParseKernelKey(pir::Operation *op);

KernelKeyTuple SaveCombineOpParseKernelKey(pir::Operation *op);

KernelKeyTuple NopOpParseKernelKey(pir::Operation *op);

KernelKeyTuple Nop_OpParseKernelKey(pir::Operation *op);

KernelKeyTuple PullGpupsSparseKernelKey(pir::Operation *op);

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ParseKernelKeyInterface)
