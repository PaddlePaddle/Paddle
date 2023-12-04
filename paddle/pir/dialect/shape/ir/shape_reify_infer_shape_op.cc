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

#include "paddle/pir/dialect/shape/ir/shape_reify_infer_shape_op.h"
#include "paddle/common/ddim.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

namespace pir::shape {

namespace {

bool DeriveShapeFromOperand(Builder *builder,
                            Value operand,
                            std::vector<Value> *reified_return_shapes) {
  auto shaped_type = operand.type().dyn_cast<ShapedTypeInterface>();
  if (!shaped_type) return false;
  reified_return_shapes->assign(
      {builder->Build<shape::ShapeOfOp>(operand).result(0)});
  return true;
}

// Returns a new scalar integer value having type `type`.
//  Here `type` must be an integer or index type.
Value MaybeCastTo(Builder &builder, Value value, Type type) {  // NOLINT
  if (type == value.type()) return value;
  if (!type.IsIndex() && !value.type().IsIndex()) {
    Value casted =
        builder.Build<shape::IndexCastOp>(builder.index_type(), value)
            .result(0);
    return builder.Build<shape::IndexCastOp>(type, casted).result(0);
  }
  return builder.Build<shape::IndexCastOp>(type, value).result(0);
}
}  // namespace

void AbsOp::Build(Builder &builder, OperationArgument &argument, Value x) {
  argument.AddInput(x);

  IrContext *ctx = IrContext::Instance();
  Type dtype = x.type().dyn_cast<ShapedTypeInterface>().GetElementType();
  pir::DDim dims = x.type().dyn_cast<DenseTensorType>().dims();
  pir::DataLayout data_layout = pir::DataLayout::NCHW;
  pir::LoD lod = {{0, 1, 2}};
  size_t offset = 0;

  argument.output_types.emplace_back(
      DenseTensorType::get(ctx, dtype, dims, data_layout, lod, offset));
}

bool AbsOp::ReifyReturnTypeShapes(Builder &builder,
                                  const std::vector<OpOperand> &operands,
                                  std::vector<Value> &reified_return_shapes) {
  return DeriveShapeFromOperand(
      &builder, operands.front().source(), &reified_return_shapes);
}

const char *TransposeOp::attributes_name[attributes_num] = {"perm"};

void TransposeOp::Build(Builder &builder,
                        OperationArgument &argument,
                        Value x,
                        std::vector<int> &perm) {
  std::vector<pir::Value> argument_inputs = {x};
  argument.AddInputs(argument_inputs);
  std::vector<pir::Attribute> vec_perm;
  for (size_t i = 0; i < static_cast<size_t>(perm.size()); i++) {
    pir::Attribute attr_perm =
        pir::Int32Attribute::get(pir::IrContext::Instance(), perm[i]);
    vec_perm.push_back(attr_perm);
  }
  pir::Attribute attr_perm =
      pir::ArrayAttribute::get(pir::IrContext::Instance(), vec_perm);
  argument.AddAttribute("perm", attr_perm);

  IrContext *ctx = IrContext::Instance();
  Type dtype = IndexType::get(ctx);
  pir::DDim in_dims = x.type().dyn_cast<DenseTensorType>().dims();
  pir::DDim out_dims = in_dims.transpose(perm);
  pir::DataLayout data_layout = pir::DataLayout::NCHW;
  pir::LoD lod = {{0, 1, 2}};
  size_t offset = 0;

  argument.output_types.emplace_back(
      DenseTensorType::get(ctx, dtype, out_dims, data_layout, lod, offset));
}

std::vector<int64_t> TransposeOp::permutation() {
  // TODO(zhangbopd): should not return just {1, 0}.
  return {1, 0};
}

bool TransposeOp::ReifyReturnTypeShapes(
    Builder &builder,
    const std::vector<OpOperand> &operands,
    std::vector<Value> &reified_return_shapes) {
  auto operand_type = operands[0].type().dyn_cast<DenseTensorType>();
  // Currently not support unranked type.
  if (!operand_type) return false;

  std::vector<int64_t> permutation = this->permutation();
  std::vector<Value> shape_values(permutation.size());

  Type shape_scalar_type = builder.index_type();

  auto to_shape_scalar_type = [&](Value v) {
    return MaybeCastTo(builder, v, shape_scalar_type);
  };

  auto shaped_type = operand_type.dyn_cast<ShapedTypeInterface>();
  auto shape_vector = shaped_type.GetDyShape();
  for (auto [idx, element] = std::tuple{0, shape_vector.begin()};
       element != shape_vector.end();
       ++idx, ++element) {
    auto it = std::find(permutation.begin(), permutation.end(), idx);
    // TODO(zhangbopd): Need BuildOrFold
    Value value_dim = to_shape_scalar_type(
        builder.Build<shape::TensorDimOp>(operands[0].source(), idx).result(0));
    shape_values[std::distance(permutation.begin(), it)] = value_dim;
  }

  Value output_shape =
      builder.Build<shape::FromElementsOp>(shape_values).result(0);
  reified_return_shapes.push_back(output_shape);

  return true;
}

void ConcatOp::Build(Builder &builder,
                     OperationArgument &argument,
                     Value x,
                     Value axis) {
  std::vector<pir::Value> argument_inputs = {x, axis};
  argument.AddInputs(argument_inputs);
}

bool ConcatOp::ReifyReturnTypeShapes(
    Builder &builder,
    const std::vector<OpOperand> &operands,
    std::vector<Value> &reified_return_shapes) {
  std::vector<Value> inputs = {x()};

  auto operand_type = inputs[0].type().dyn_cast<DenseTensorType>();
  // Currently not support unranked type.
  if (!operand_type) return false;

  Type shapeScalarType = builder.index_type();
  auto to_shape_scalar_type = [&](Value v) {
    return MaybeCastTo(builder, v, shapeScalarType);
  };

  std::vector<std::vector<Value>> all_shape_values;
  for (size_t inputId = 0; inputId < inputs.size(); ++inputId) {
    Value operand = inputs[inputId];
    auto operand_type = operand.type().dyn_cast<DenseTensorType>();
    if (!operand_type) return false;

    std::vector<Value> shape_values;

    auto shaped_type = operand_type.dyn_cast<ShapedTypeInterface>();
    auto shape_vector = shaped_type.GetDyShape();
    for (auto [idx, element] = std::tuple{0, shape_vector.begin()};
         element != shape_vector.end();
         ++idx, ++element) {
      Value value_dim = to_shape_scalar_type(
          builder.Build<shape::TensorDimOp>(operand, idx).result(0));
      shape_values.push_back(value_dim);
    }
    all_shape_values.emplace_back(std::move(shape_values));
  }

  [[maybe_unused]] int axis = this->dimension();
  auto &shape_values = all_shape_values[0];
  for (size_t vecId = 1; vecId < all_shape_values.size(); ++vecId) {
    auto &otherShapeValues = all_shape_values[vecId];
    if (otherShapeValues.size() != shape_values.size()) return false;
    // TODO(zhangbopd): AddIOp
    // shape_values[axis] =
    //     builder.Build<arith::AddIOp>(shape_values[axis],
    //     otherShapeValues[axis]);
  }

  Value output_shape =
      builder.Build<shape::FromElementsOp>(shape_values).result(0);
  reified_return_shapes.push_back(output_shape);
  return true;
}

}  // namespace pir::shape

IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::AbsOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::TransposeOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::ConcatOp)
