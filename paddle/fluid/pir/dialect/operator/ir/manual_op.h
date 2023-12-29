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
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/decomp.h"
#include "paddle/fluid/pir/dialect/operator/interface/get_kernel_type_for_var.h"
#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/vjp.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/operation_utils.h"

namespace paddle {
namespace dialect {

class AddNOp : public pir::Op<AddNOp,
                              paddle::dialect::OpYamlInfoInterface,
                              paddle::dialect::InferMetaInterface,
                              paddle::dialect::VjpInterface,
                              paddle::dialect::DecompInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.add_n"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value inputs);

  void VerifySig();
  pir::Value inputs() { return operand_source(0); }
  pir::OpResult out() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
  static std::vector<std::vector<pir::OpResult>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::Value>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);
  static std::vector<std::vector<pir::OpResult>> Decomp(pir::Operation *op);
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
                    pir::Value inputs_);

  void VerifySig();
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
                    pir::Value inputs_);

  void VerifySig();
  pir::Value inputs() { return operand_source(0); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class FusedGemmEpilogueOp
    : public pir::Op<FusedGemmEpilogueOp,
                     paddle::dialect::OpYamlInfoInterface,
                     paddle::dialect::InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.fused_gemm_epilogue"; }
  static const char *attributes_name[3];
  static constexpr uint32_t attributes_num = 3;
  static OpInfoTuple GetOpInfo();

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,
                    pir::Value y_,
                    pir::Value bias_,
                    pir::AttributeMap attributes);
  void VerifySig();
  pir::Value x() { return operand_source(0); }
  pir::Value y() { return operand_source(1); }
  pir::Value bias() { return operand_source(2); }
  pir::OpResult out() { return result(0); }
  pir::OpResult reserve_space() { return result(1); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class FusedGemmEpilogueGradOp
    : public pir::Op<FusedGemmEpilogueGradOp,
                     paddle::dialect::OpYamlInfoInterface,
                     paddle::dialect::InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.fused_gemm_epilogue_grad"; }
  static const char *attributes_name[3];
  static constexpr uint32_t attributes_num = 3;
  static OpInfoTuple GetOpInfo();

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,
                    pir::Value y_,
                    pir::Value reserve_space_,
                    pir::Value out_grad_,
                    pir::AttributeMap attributes);
  void VerifySig();
  pir::Value x() { return operand_source(0); }
  pir::Value y() { return operand_source(1); }
  pir::Value reserve_space() { return operand_source(2); }
  pir::Value out_grad() { return operand_source(3); }
  pir::OpResult x_grad() { return result(0); }
  pir::OpResult y_grad() { return result(1); }
  pir::OpResult bias_grad() { return result(2); }

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
                    pir::Value x_,
                    float axis = 0);
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value out_grad_,
                    pir::Value axis_);

  void VerifySig();
  pir::Value out_grad() { return operand_source(0); }
  pir::Value axis() { return operand_source(1); }
  pir::OpResult x_grad() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class CreateArrayOp
    : public pir::Op<CreateArrayOp, OpYamlInfoInterface, InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.create_array"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    phi::DataType dtype = phi::DataType::FLOAT32);
  void VerifySig();
  pir::OpResult out() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class ArrayLengthOp
    : public pir::Op<ArrayLengthOp, OpYamlInfoInterface, InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.array_length"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x);
  void VerifySig();
  pir::Value x() { return operand_source(0); }
  pir::OpResult out() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class ArrayReadOp : public pir::Op<ArrayReadOp,
                                   OpYamlInfoInterface,
                                   paddle::dialect::VjpInterface,
                                   InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.array_read"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value array,
                    int64_t i);
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value array,
                    pir::Value i);
  void VerifySig();
  pir::Value array() { return operand_source(0); }
  pir::Value i() { return operand_source(1); }
  pir::OpResult out() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
  static std::vector<std::vector<pir::OpResult>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::Value>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);
};

class ArrayWrite_Op : public pir::Op<ArrayWrite_Op,
                                     OpYamlInfoInterface,
                                     paddle::dialect::VjpInterface,
                                     InferMetaInterface,
                                     InplaceTrait> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.array_write_"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value array,
                    pir::Value x,
                    pir::Value i);
  void VerifySig();
  pir::Value array() { return operand_source(0); }
  pir::Value x() { return operand_source(1); }
  pir::Value i() { return operand_source(2); }
  pir::OpResult out() { return result(0); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
  static std::vector<std::vector<pir::OpResult>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::Value>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);
};

class ArrayToTensorOp
    : public pir::Op<ArrayToTensorOp, OpYamlInfoInterface, InferMetaInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.array_to_tensor"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x,
                    int axis,
                    bool use_stack);
  void VerifySig();
  pir::Value x() { return operand_source(0); }
  pir::OpResult out() { return result(0); }
  pir::OpResult out_index() { return result(2); }
  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class SliceArrayOp
    : public pir::Op<SliceArrayOp,
                     paddle::dialect::OpYamlInfoInterface,
                     paddle::dialect::InferMetaInterface,
                     paddle::dialect::GetKernelTypeForVarInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.slice_array"; }
  static const char *attributes_name[2];
  static constexpr uint32_t attributes_num = 2;
  static OpInfoTuple GetOpInfo();

  void VerifySig();

  static phi::DataType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DataType &tensor_dtype,
      const phi::DataType &expected_kernel_dtype);

  pir::Value input() { return operand_source(0); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class SliceArrayDenseOp
    : public pir::Op<SliceArrayDenseOp,
                     paddle::dialect::OpYamlInfoInterface,
                     paddle::dialect::InferMetaInterface,
                     paddle::dialect::GetKernelTypeForVarInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.slice_array_dense"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static OpInfoTuple GetOpInfo();
  void VerifySig();

  static phi::DataType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DataType &tensor_dtype,
      const phi::DataType &expected_kernel_dtype);

  pir::Value input() { return operand_source(0); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class AssignArray_Op
    : public pir::Op<AssignArray_Op,
                     paddle::dialect::OpYamlInfoInterface,
                     paddle::dialect::InferMetaInterface,
                     paddle::dialect::GetKernelTypeForVarInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.assign_array_"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static OpInfoTuple GetOpInfo();

  void VerifySig();

  static phi::DataType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DataType &tensor_dtype,
      const phi::DataType &expected_kernel_dtype);

  pir::Value x() { return operand_source(0); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
};

class ExpandOp : public pir::Op<ExpandOp,
                                paddle::dialect::OpYamlInfoInterface,
                                paddle::dialect::InferMetaInterface,
                                paddle::dialect::VjpInterface,
                                paddle::dialect::GetKernelTypeForVarInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.expand"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,                     // NOLINT
                    const std::vector<int64_t> &shape = {});
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,                     // NOLINT
                    pir::Value shape_                  // NOLINT
  );
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,                     // NOLINT
                    pir::AttributeMap attributes       // NOLINT
  );

  void VerifySig();

  static phi::DataType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DataType &tensor_dtype,
      const phi::DataType &expected_kernel_dtype);

  pir::Value x() { return operand_source(0); }
  pir::Value shape() { return operand_source(1); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
  static std::vector<std::vector<pir::OpResult>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::Value>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);
};

class SelectInputOp : public pir::Op<SelectInputOp> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.select_input"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  void VerifySig();
  pir::Value mask() { return operand_source(0); }
  pir::OpResult out() { return result(0); }
};

class IncrementOp
    : public pir::Op<IncrementOp,
                     paddle::dialect::OpYamlInfoInterface,
                     paddle::dialect::InferMetaInterface,
                     paddle::dialect::VjpInterface,
                     paddle::dialect::GetKernelTypeForVarInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.increment"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,                     // NOLINT
                    float value = 1.0);

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,                     // NOLINT
                    pir::AttributeMap attributes);

  void VerifySig();

  static phi::DataType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DataType &tensor_dtype,
      const phi::DataType &expected_kernel_dtype);

  pir::Value x() { return operand_source(0); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
  static std::vector<std::vector<pir::OpResult>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::Value>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);
};

class Increment_Op
    : public pir::Op<Increment_Op,
                     paddle::dialect::OpYamlInfoInterface,
                     paddle::dialect::InferMetaInterface,
                     paddle::dialect::VjpInterface,
                     paddle::dialect::GetKernelTypeForVarInterface,
                     paddle::dialect::InplaceTrait> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.increment_"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  static OpInfoTuple GetOpInfo();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,                     // NOLINT
                    float value = 1.0);

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value x_,                     // NOLINT
                    pir::AttributeMap attributes);

  void VerifySig();

  static phi::DataType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DataType &tensor_dtype,
      const phi::DataType &expected_kernel_dtype);

  pir::Value x() { return operand_source(0); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);
  static std::vector<std::vector<pir::OpResult>> Vjp(
      pir::Operation *op,
      const std::vector<std::vector<pir::Value>> &inputs_,
      const std::vector<std::vector<pir::Value>> &outputs,
      const std::vector<std::vector<pir::Value>> &out_grads,
      const std::vector<std::vector<bool>> &stop_gradients);
};

class IR_API ShapeBroadcastOp : public pir::Op<ShapeBroadcastOp,paddle::dialect::InferMetaInterface,paddle::dialect::GetKernelTypeForVarInterface> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.shape_broadcast"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static void Build(pir::Builder &builder, pir::OperationArgument &argument, pir::Value x_, pir::Value y_);
  
  void VerifySig() {}


  pir::Value x() { return operand_source(0); }
  pir::Value y() { return operand_source(1); }
  pir::OpResult out() { return result(0); }

  static void InferMeta(phi::InferMetaContext *infer_meta);

  static phi::DataType GetKernelTypeForVar(
      const std::string& var_name,
        const phi::DataType& tensor_dtype,
        const phi::DataType& expected_kernel_dtype);

};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AddNOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SplitGradOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AddN_Op)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AddNWithKernelOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::FusedGemmEpilogueOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::FusedGemmEpilogueGradOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::CreateArrayOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayLengthOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayReadOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayWrite_Op)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SliceArrayOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SliceArrayDenseOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AssignArray_Op)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ArrayToTensorOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ExpandOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SelectInputOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::IncrementOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::Increment_Op)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ShapeBroadcastOp)