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

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/platform/tensorrt/engine_params.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"

namespace paddle {
namespace dialect {

class TensorRTEngineOp
    : public pir::Op<TensorRTEngineOp, paddle::dialect::OpYamlInfoInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.tensorrt_engine"; }
  static const char *attributes_name[13];
  static constexpr uint32_t attributes_num = 13;
  static OpInfoTuple GetOpInfo();

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x,
                    paddle::platform::EngineParams trt_params,
                    std::vector<std::string> input_names,
                    std::vector<std::string> output_names,
                    std::vector<std::vector<int64_t>> outputs_shape,
                    std::vector<phi::DataType> outputs_dtype,
                    const std::string &converter_debug_info);

  void VerifySig();

  pir::Value x() { return operand_source(0); }
  pir::Value out() { return result(0); }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::TensorRTEngineOp)
