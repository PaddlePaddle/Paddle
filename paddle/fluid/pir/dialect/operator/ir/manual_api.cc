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

#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/parameter.h"
namespace paddle {
namespace dialect {

pir::OpResult builtin_combine(const std::vector<pir::Value>& x) {
  auto combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(x);
  return combine_op.out();
}

std::vector<pir::OpResult> add_n_grad(const std::vector<pir::Value>& inputs,
                                      const pir::Value& out_grad) {
  std::vector<pir::OpResult> inputs_grad;
  for (size_t i = 0; i < inputs.size(); i++) {
    paddle::dialect::ScaleOp scale_op =
        ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ScaleOp>(
            out_grad, 1.0, 0.0, true);
    inputs_grad.push_back(scale_op.result(0));
  }
  return inputs_grad;
}

pir::OpResult zeros_like(const pir::Value& x,
                         const phi::DataType dtype,
                         const Place& place) {
  return paddle::dialect::full_like(x, 0, dtype, place);
}

pir::OpResult parameter(const std::string& name) {
  pir::Parameter* param = ApiBuilder::Instance().GetParameter(name);
  pir::ParameterOp parameter_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::ParameterOp>(
          name, param->type());
  return parameter_op.result(0);
}

void set_parameter(const pir::Value& parameter, const std::string& name) {
  std::unique_ptr<pir::Parameter> param(
      new pir::Parameter(nullptr, 0, parameter.type()));
  ApiBuilder::Instance().SetParameter(name, std::move(param));
  ApiBuilder::Instance().GetBuilder()->Build<pir::SetParameterOp>(parameter,
                                                                  name);
}

pir::OpResult embedding_grad(const pir::Value& x,
                             const pir::Value& weight,
                             const pir::Value& out_grad,
                             int64_t padding_idx,
                             bool sparse) {
  if (weight.type().isa<paddle::dialect::DenseTensorType>()) {
    if (sparse) {
      auto embedding_grad_op =
          ApiBuilder::Instance()
              .GetBuilder()
              ->Build<paddle::dialect::EmbeddingSparseGradOp>(
                  x, weight, out_grad, padding_idx);
      return embedding_grad_op.weight_grad();
    } else {
      auto embedding_grad_op = ApiBuilder::Instance()
                                   .GetBuilder()
                                   ->Build<paddle::dialect::EmbeddingGradOp>(
                                       x, weight, out_grad, padding_idx);
      return embedding_grad_op.weight_grad();
    }
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Now we do not support sparse weight embedding_grad."));
  }
}

pir::OpResult split_with_num_grad(const std::vector<pir::Value>& out_grad,
                                  int axis) {
  auto out_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grad);
  paddle::dialect::SplitGradOp split_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          out_grad_combine_op.out(), axis);
  return split_grad_op.result(0);
}

pir::OpResult split_with_num_grad(const std::vector<pir::Value>& out_grad,
                                  const pir::Value& axis) {
  auto out_grad_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(out_grad);
  paddle::dialect::SplitGradOp split_grad_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::SplitGradOp>(
          out_grad_combine_op.out(), axis);
  return split_grad_op.result(0);
}

pir::OpResult ones(const std::vector<int64_t>& shape,
                   phi::DataType dtype,
                   const Place& place) {
  return paddle::dialect::full(shape, 1, dtype, place);
}

pir::OpResult ones_like(pir::Value x_,
                        phi::DataType dtype,
                        const Place& place) {
  return paddle::dialect::full_like(x_, 1, dtype, place);
}

pir::OpResult zeros(const std::vector<int64_t>& shape,
                    phi::DataType dtype,
                    const Place& place) {
  return paddle::dialect::full(shape, 0, dtype, place);
}

pir::OpResult create_array(phi::DataType dtype) {
  auto create_array_op = ApiBuilder::Instance()
                             .GetBuilder()
                             ->Build<paddle::dialect::CreateArrayOp>(dtype);
  return create_array_op.out();
}

pir::OpResult array_length(pir::Value x) {
  auto array_length_op = ApiBuilder::Instance()
                             .GetBuilder()
                             ->Build<paddle::dialect::ArrayLengthOp>(x);
  return array_length_op.out();
}

pir::OpResult array_read(pir::Value array, pir::Value i) {
  auto array_read_op =
      ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::ArrayReadOp>(
          array, i);
  return array_read_op.out();
}

pir::OpResult array_write_(pir::Value array, pir::Value x, pir::Value i) {
  auto array_write_op =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ArrayWrite_Op>(array, x, i);
  return array_write_op.out();
}

std::tuple<pir::OpResult, pir::OpResult> array_to_tensor(pir::Value x,
                                                         int axis,
                                                         bool use_stack) {
  auto array_to_tensor =
      ApiBuilder::Instance()
          .GetBuilder()
          ->Build<paddle::dialect::ArrayToTensorOp>(x, axis, use_stack);
  return std::make_tuple(array_to_tensor.result(0), array_to_tensor.result(1));
}

}  // namespace dialect
}  // namespace paddle
