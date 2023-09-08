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
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/operation_utils.h"

namespace paddle {
namespace dialect {

class AddNOp : public pir::Op<AddNOp, OpYamlInfoInterface, InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.add_n"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::OpResult inputs);

  void Verify();
  pir::Value inputs() { return operand_source(0); }
  pir::OpResult out() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class AddN_Op : public pir::Op<AddN_Op,
                               paddle::dialect::OpYamlInfoInterface,
                               paddle::dialect::InferMetaInterface,
                               paddle::dialect::InplaceTrait> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.add_n_"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::OpResult inputs_);

  void Verify();
  pir::Value inputs() { return operand_source(0); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class AddNWithKernelOp : public pir::Op<AddNWithKernelOp,
                                        paddle::dialect::OpYamlInfoInterface,
                                        paddle::dialect::InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.add_n_with_kernel"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::OpResult inputs_);

  void Verify();
  pir::Value inputs() { return operand_source(0); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class SplitGradOp : public pir::Op<SplitGradOp, OpYamlInfoInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.split_grad"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::OpResult x_,
                    float axis = 0);
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::OpResult out_grad_,
                    pir::OpResult axis_);

  void Verify();
  pir::Value out_grad() { return operand_source(0); }
  pir::Value axis() { return operand_source(1); }
  pir::OpResult x_grad() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AddNOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SplitGradOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AddN_Op)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AddNWithKernelOp)

#endif
