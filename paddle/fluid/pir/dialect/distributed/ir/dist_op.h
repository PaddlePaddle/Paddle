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
#include "paddle/fluid/pir/dialect/operator/interface/vjp.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/operation_utils.h"

namespace paddle {
namespace dialect {
class TensorDistAttribute;

class ShardTensorOp : public pir::Op<ShardTensorOp> {
 public:
  using Op::Op;
  static const char* name() { return "dist_op.shard_tensor"; }
  static const char* attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  TEST_API static void Build(pir::Builder& builder,             // NOLINT
                             pir::OperationArgument& argument,  // NOLINT
                             pir::Value input,
                             pir::AttributeMap attributes);
  pir::Value input() { return operand_source(0); }
  pir::Value out() { return result(0); }
  void VerifySig();
};

class ReshardOp : public pir::Op<ReshardOp, VjpInterface, OpYamlInfoInterface> {
 public:
  using Op::Op;
  static const char* name() { return "dist_op.reshard"; }
  static const char* attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  TEST_API static void Build(pir::Builder& builder,             // NOLINT
                             pir::OperationArgument& argument,  // NOLINT
                             pir::Value input,
                             TensorDistAttribute tensor_dist_attr);

  static OpInfoTuple GetOpInfo();
  static std::vector<std::vector<pir::Value>> Vjp(
      pir::Operation* op,
      const std::vector<std::vector<pir::Value>>& inputs_,
      const std::vector<std::vector<pir::Value>>& outputs,
      const std::vector<std::vector<pir::Value>>& out_grads,
      const std::vector<std::vector<bool>>& stop_gradients);

  void VerifySig();
};

class LocalTensorsFromDistOp
    : public pir::Op<LocalTensorsFromDistOp, VjpInterface> {
 public:
  using Op::Op;
  static const char* name() { return "dist_op.local_tensors_from_dtensor"; }
  static const char* attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  TEST_API static void Build(
      pir::Builder& builder,             // NOLINT
      pir::OperationArgument& argument,  // NOLINT
      pir::Value input,
      const std::vector<TensorDistAttribute>& local_dist_attrs,
      const TensorDistAttribute& global_dist_attr);

  static OpInfoTuple GetOpInfo();
  static std::vector<std::vector<pir::Value>> Vjp(
      pir::Operation* op,
      const std::vector<std::vector<pir::Value>>& inputs_,
      const std::vector<std::vector<pir::Value>>& outputs,
      const std::vector<std::vector<pir::Value>>& out_grads,
      const std::vector<std::vector<bool>>& stop_gradients);

  void VerifySig();
  std::vector<pir::Value> results() { return operation()->results(); }
};

class DistTensorFromLocalsOp
    : public pir::Op<DistTensorFromLocalsOp, VjpInterface> {
 public:
  using Op::Op;
  static const char* name() { return "dist_op.dtensor_from_local_tensors"; }
  static const char* attributes_name[1];
  static constexpr uint32_t attributes_num = 1;
  TEST_API static void Build(
      pir::Builder& builder,             // NOLINT
      pir::OperationArgument& argument,  // NOLINT
      const std::vector<pir::Value>& inputs,
      const std::vector<TensorDistAttribute>& local_dist_attrs,
      const TensorDistAttribute& global_dist_attr,
      const phi::DDim& global_dims);

  static OpInfoTuple GetOpInfo();
  static std::vector<std::vector<pir::Value>> Vjp(
      pir::Operation* op,
      const std::vector<std::vector<pir::Value>>& inputs_,
      const std::vector<std::vector<pir::Value>>& outputs,
      const std::vector<std::vector<pir::Value>>& out_grads,
      const std::vector<std::vector<bool>>& stop_gradients);

  void VerifySig();
  std::vector<pir::Value> results() { return operation()->results(); }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ShardTensorOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ReshardOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::LocalTensorsFromDistOp)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DistTensorFromLocalsOp)
