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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/primitive/composite/composite.h"
#include "paddle/fluid/primitive/rule/vjp/details.h"
#include "paddle/fluid/primitive/rule/vjp/generated/generated_vjp.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_base.h"

// TODO(chenzhuo)
// this file will be generated in pd_op_decomp_vjp.cc

namespace paddle {
namespace dialect {
using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<pir::Value>> StackGradOp::DecompVjp(
    pir::Operation* op) {
  VLOG(4) << "Decomp call stack_grad's decomp interface begin";

  StackGradOp op_obj = op->dyn_cast<StackGradOp>();
  (void)op_obj;

  FLAGS_tensor_operants_mode = "static";

  VLOG(6) << "Decomp Prepare inputs of stack_grad";

  pir::CombineOp combine_op_obj_x =
      op_obj.x().defining_op()->dyn_cast<pir::CombineOp>();
  std::vector<Tensor> x;
  for (size_t idx = 0; idx < combine_op_obj_x.inputs().size(); idx++) {
    x.emplace_back(std::make_shared<primitive::LazyTensor>(
        combine_op_obj_x.inputs()[idx]));
  }
  Tensor out_grad(std::make_shared<primitive::LazyTensor>(op_obj.out_grad()));

  VLOG(6) << "Decomp prepare attributes of stack_grad";
  int axis = op->attribute("axis").dyn_cast<pir::Int32Attribute>().data();

  VLOG(6) << "Decomp call stack_grad's backward composite rule prepare";

  std::vector<std::vector<bool>> stop_gradients(op->results().size());
  if (combine_op_obj_x->HasAttribute(kAttrStopGradients)) {
    auto stop_gradients_attr = op->attribute(kAttrStopGradients)
                                   .dyn_cast<pir::ArrayAttribute>()
                                   .AsVector();
    for (size_t i = 0; i < stop_gradients[0].size(); ++i) {
      stop_gradients[0].push_back(
          stop_gradients_attr[i].dyn_cast<pir::BoolAttribute>().data());
    }

    VLOG(4) << " stop_gradients is set ";
  } else {
    std::vector<bool> x_grad_stop_gradient(combine_op_obj_x.inputs().size(),
                                           false);
    stop_gradients[0] = x_grad_stop_gradient;
    VLOG(4) << " stop_gradients is not set ";
  }

  std::vector<std::vector<paddle::Tensor>> tensor_res;
  for (auto arg : stop_gradients) {
    tensor_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "stack_grad";
  FLAGS_tensor_operants_mode = "static";
  VLOG(4) << "Call Pir Decomposed backward op stack_grad";

  std::vector<paddle::Tensor*> x_grad(stop_gradients[0].size(), nullptr);
  for (size_t i = 0; i < stop_gradients[0].size(); i++) {
    x_grad[i] = !stop_gradients[0][i] ? &tensor_res[0][i] : nullptr;
  }

  paddle::primitive::details::stack_grad<primitive::LazyTensor>(
      x, out_grad, axis, x_grad);
  std::vector<std::vector<pir::Value>> res(tensor_res.size());

  for (size_t i = 0; i < tensor_res.size(); ++i) {
    res[i].resize(tensor_res[i].size());
    for (size_t j = 0; j < tensor_res[i].size(); ++j) {
      if (tensor_res[i][j].defined()) {
        res[i][j] = std::static_pointer_cast<primitive::LazyTensor>(
                        tensor_res[i][j].impl())
                        ->value();
      }
    }
  }

  VLOG(4) << "Decomp call stack_grad's decomp interface end";
  return res;
}

std::vector<std::vector<pir::Value>> ConcatGradOp::DecompVjp(
    pir::Operation* op) {
  VLOG(4) << "Decomp call concat_grad's decomp interface begin";

  ConcatGradOp op_obj = op->dyn_cast<ConcatGradOp>();
  (void)op_obj;

  FLAGS_tensor_operants_mode = "static";

  VLOG(6) << "Decomp Prepare inputs of concat_grad";

  pir::CombineOp combine_op_obj_x =
      op_obj.x().defining_op()->dyn_cast<pir::CombineOp>();
  std::vector<Tensor> x;
  for (size_t idx = 0; idx < combine_op_obj_x.inputs().size(); idx++) {
    x.emplace_back(std::make_shared<primitive::LazyTensor>(
        combine_op_obj_x.inputs()[idx]));
  }
  Tensor out_grad(std::make_shared<primitive::LazyTensor>(op_obj.out_grad()));

  VLOG(6) << "Decomp prepare attributes of concat_grad";

  Tensor axis_(std::make_shared<primitive::LazyTensor>(op_obj.axis()));

  auto* axis_define_op =
      std::static_pointer_cast<primitive::LazyTensor>(axis_.impl())
          ->value()
          .defining_op();
  if (axis_define_op->name() != "pd_op.full") {
    PADDLE_THROW(platform::errors::Unimplemented(
        "We don't support dynamic tensors "
        "attribute axis for concat_grad decomposition "
        "for now. "));
  }
  Scalar axis = axis_define_op->attribute("value")
                    .dyn_cast<paddle::dialect::ScalarAttribute>()
                    .data();

  VLOG(6) << "Decomp call concat_grad's backward composite rule prepare";

  std::vector<std::vector<bool>> stop_gradients(op->results().size());
  if (combine_op_obj_x->HasAttribute(kAttrStopGradients)) {
    auto stop_gradients_attr = op->attribute(kAttrStopGradients)
                                   .dyn_cast<pir::ArrayAttribute>()
                                   .AsVector();
    for (size_t i = 0; i < stop_gradients[0].size(); ++i) {
      stop_gradients[0].push_back(
          stop_gradients_attr[i].dyn_cast<pir::BoolAttribute>().data());
    }

    VLOG(4) << " stop_gradients is set ";
  } else {
    std::vector<bool> x_grad_stop_gradient(combine_op_obj_x.inputs().size(),
                                           false);
    stop_gradients[0] = x_grad_stop_gradient;
    VLOG(4) << " stop_gradients is not set ";
  }

  std::vector<std::vector<paddle::Tensor>> tensor_res;
  for (auto arg : stop_gradients) {
    tensor_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "concat_grad";
  FLAGS_tensor_operants_mode = "static";
  VLOG(4) << "Call Pir Decomposed backward op concat_grad";

  std::vector<paddle::Tensor*> x_grad(stop_gradients[0].size(), nullptr);
  for (size_t i = 0; i < stop_gradients[0].size(); i++) {
    x_grad[i] = !stop_gradients[0][i] ? &tensor_res[0][i] : nullptr;
  }

  paddle::primitive::details::concat_grad<primitive::LazyTensor>(
      x, out_grad, axis, x_grad);
  std::vector<std::vector<pir::Value>> res(tensor_res.size());

  for (size_t i = 0; i < tensor_res.size(); ++i) {
    res[i].resize(tensor_res[i].size());
    for (size_t j = 0; j < tensor_res[i].size(); ++j) {
      if (tensor_res[i][j].defined()) {
        res[i][j] = std::static_pointer_cast<primitive::LazyTensor>(
                        tensor_res[i][j].impl())
                        ->value();
      }
    }
  }

  VLOG(4) << "Decomp call concat_grad's decomp interface end";
  return res;
}

}  // namespace dialect
}  // namespace paddle
