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

#ifdef GET_MANUAL_OP_LIST
#undef GET_MANUAL_OP_LIST
paddle::dialect::AddNOp, paddle::dialect::SplitGradOp

#else

#pragma once
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/interface/infermeta.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/interface/op_yaml_info.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/op_yaml_info_util.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/utils.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/op_base.h"
#include "paddle/ir/core/operation_utils.h"
#include "paddle/phi/core/infermeta_utils.h"

namespace paddle {
namespace dialect {

class AddNOp : public ir::Op<AddNOp, OpYamlInfoInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd.add_n"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static OpInfoTuple GetOpInfo();
  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    ir::OpResult inputs);

  void Verify();
  ir::Value inputs() { return operand_source(0); }
  ir::OpResult out() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class SplitGradOp : public ir::Op<SplitGradOp, OpYamlInfoInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd.split_grad"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  static OpInfoTuple GetOpInfo();
  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    ir::OpResult x_,
                    float axis = 0);
  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    ir::OpResult out_grad_,
                    ir::OpResult axis_);

  void Verify();
  ir::Value out_grad() { return operand_source(0); }
  ir::Value axis() { return operand_source(1); }
  ir::OpResult x_grad() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AddNOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SplitGradOp)

#endif
