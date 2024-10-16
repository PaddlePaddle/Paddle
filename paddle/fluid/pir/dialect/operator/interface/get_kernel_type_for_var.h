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

#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace dialect {
class GetKernelTypeForVarInterface
    : public pir::OpInterfaceBase<GetKernelTypeForVarInterface> {
 public:
  struct Concept {
    explicit Concept(phi::DataType (*get_kernel_type_for_var)(
        const std::string& var_name,
        const phi::DataType& tensor_dtype,
        const phi::DataType& expected_kernel_dtype))
        : get_kernel_type_for_var_(get_kernel_type_for_var) {}

    phi::DataType (*get_kernel_type_for_var_)(
        const std::string& var_name,
        const phi::DataType& tensor_dtype,
        const phi::DataType& expected_kernel_dtype);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static phi::DataType GetKernelTypeForVar(
        const std::string& var_name,
        const phi::DataType& tensor_dtype,
        const phi::DataType& expected_kernel_dtype) {
      return ConcreteOp::GetKernelTypeForVar(
          var_name, tensor_dtype, expected_kernel_dtype);
    }

    Model() : Concept(GetKernelTypeForVar) {}
  };

  /// Constructor
  GetKernelTypeForVarInterface(const pir::Operation* op, Concept* impl)
      : pir::OpInterfaceBase<GetKernelTypeForVarInterface>(op), impl_(impl) {}

  phi::DataType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DataType& tensor_dtype,
      const phi::DataType& expected_kernel_dtype) {
    return impl_->get_kernel_type_for_var_(
        var_name, tensor_dtype, expected_kernel_dtype);
  }

 private:
  Concept* impl_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::GetKernelTypeForVarInterface)
