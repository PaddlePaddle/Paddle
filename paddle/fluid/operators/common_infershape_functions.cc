/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/common_infershape_functions.h"

// This file almostly contains all the infershape functions that are used in
// operators.

namespace paddle {
namespace operators {

// shape input(0) -> output(0) without change.
void UnchagedInferShape(framework::InferShapeContext* ctx) {
  ctx->ShareDim(ctx->GetInputNameByIdx(0), /*->*/ ctx->GetOutputNameByIdx(0));
  ctx->ShareLoD(ctx->GetInputNameByIdx(0), /*->*/ ctx->GetOutputNameByIdx(0));
}  // namespace operators

// broadcast input(0) and input(1) -> output(0)
void BinaryOpBroadcastInferShape(framework::InferShapeContext* ctx) {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BinaryOp");
  OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "BinaryOp");
  //   OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ElementwiseOp");

  //   PADDLE_ENFORCE_EQ(
  //       ctx->GetInputsVarType("Y").front(),
  //       framework::proto::VarType::LOD_TENSOR,
  //       platform::errors::InvalidArgument(
  //           "The input var's type should be LoDTensor, but the "
  //           "received is %s [%s].",
  //           ctx->GetInputsVarType("Y").front(), ctx->Inputs("Y").front()));

  //   if (ctx->GetInputsVarType("X").front() ==
  //       framework::proto::VarType::SELECTED_ROWS) {
  //     PADDLE_ENFORCE_EQ(
  //         ctx->GetInputDim("Y").size(), 1u,
  //         platform::errors::InvalidArgument(
  //             "For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
  //             "), Y must be scalar, the size of Y should be 1. "
  //             "But reveived the size of Y = %s.",
  //             ctx->GetInputDim("Y").size()));
  //     PADDLE_ENFORCE_EQ(
  //         ctx->GetInputDim("Y")[0], 1,
  //         platform::errors::InvalidArgument(
  //             "For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
  //             "), Y must be scalar, the first dimension of Y should be 1. "
  //             "But reveived the first dimension of Y = %s.",
  //             ctx->GetInputDim("Y")[0]));
  //   } else if (ctx->GetInputsVarType("X").front() !=
  //              framework::proto::VarType::LOD_TENSOR) {
  //     PADDLE_THROW(platform::errors::InvalidArgument(
  //         "Input X's type[%s] is not supported by elementwise_op. Please set
  //         " "its type to LOD_TENSOR.", ctx->GetInputsVarType("X").front()));
  //   }

  //   if (ctx->GetInputDim("X") == ctx->GetInputDim("Y")) {
  //     ctx->ShareDim("X", /*->*/ "Out");
  //     ctx->ShareLoD("X", /*->*/ "Out");
  //   } else {
  //     auto x_dims = ctx->GetInputDim("X");
  //     auto y_dims = ctx->GetInputDim("Y");
  //     int max_dim = std::max(x_dims.size(), y_dims.size());
  //     int axis = ctx->Attrs().Get<int>("axis");
  //     axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  //     std::vector<int> x_dims_array(max_dim);
  //     std::vector<int> y_dims_array(max_dim);
  //     std::vector<int> out_dims_array(max_dim);
  //     GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
  //                            y_dims_array.data(), out_dims_array.data(),
  //                            max_dim, axis);
  //     ctx->SetOutputDim("Out", framework::make_ddim(out_dims_array));
  //     // to do
  //     ctx->ShareLoD("X", /*->*/ "Out");
  //   }
}

}  // namespace operators
}  // namespace paddle
