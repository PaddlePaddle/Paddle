// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/decomp.h"
#include "paddle/fluid/pir/dialect/operator/interface/get_kernel_type_for_var.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/parse_kernel_key.h"
#include "paddle/fluid/pir/dialect/operator/interface/vjp.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/custom_vjp.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"

namespace paddle {
namespace onednn {
namespace dialect {

class ExpandOp : public pir::Op<ExpandOp,
                                paddle::dialect::OpYamlInfoInterface,
                                paddle::dialect::InferMetaInterface,
                                paddle::dialect::GetKernelTypeForVarInterface,
                                paddle::dialect::OneDNNTrait> {
 public:
  using Op::Op;
  static const char* name() { return "onednn_op.expand"; }
  static const char* attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder& builder,             // NOLINT
                    pir::OperationArgument& argument,  // NOLINT
                    pir::Value x_,
                    const std::vector<int64_t>& shape = {},
                    const std::string& mkldnn_data_type = "float32");
  static void Build(pir::Builder& builder,             // NOLINT
                    pir::OperationArgument& argument,  // NOLINT
                    pir::Value x_,
                    pir::Value shape_,
                    const std::string& mkldnn_data_type = "float32");
  static void Build(pir::Builder& builder,             // NOLINT
                    pir::OperationArgument& argument,  // NOLINT
                    pir::Value x_,
                    pir::AttributeMap attributes);

  void VerifySig();

  static phi::DataType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DataType& tensor_dtype,
      const phi::DataType& expected_kernel_dtype);

  pir::Value x() { return operand_source(0); }
  pir::Value shape() { return operand_source(1); }
  pir::Value out() { return result(0); }

  static void InferMeta(phi::InferMetaContext* infer_meta);
  static std::vector<pir::Type> InferMeta(
      const std::vector<pir::Value>& input_values,
      pir::AttributeMap* p_attributes);  // NOLINT
};

}  // namespace dialect
}  // namespace onednn
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::onednn::dialect::ExpandOp)
