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
#include "paddle/fluid/primitive/base/lazy_tensor.h"
#include "paddle/fluid/primitive/decomp_rule/decomp_rule/composite.h"
#include "paddle/fluid/primitive/decomp_rule/decomp_vjp/details.h"
#include "paddle/fluid/primitive/vjp_interface/generated/generated_vjp.h"
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
    for (size_t i = 0; i < stop_gradients_attr.size(); ++i) {
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
    PADDLE_THROW(common::errors::Unimplemented(
        "We don't support dynamic tensors "
        "attribute axis for concat_grad decomposition "
        "for now. "));
  }
  Scalar axis = axis_define_op->attribute("value")
                    .dyn_cast<paddle::dialect::ScalarAttribute>()
                    .data();

  VLOG(4) << "Decomp call concat_grad's backward composite rule prepare";

  std::vector<std::vector<bool>> stop_gradients(op->results().size());
  auto splitop = op->results()[0].first_use().owner();

  if (splitop->HasAttribute("current_bwd_op_stop_gradients")) {
    auto stop_gradients_attr =
        splitop->attribute("current_bwd_op_stop_gradients")
            .dyn_cast<pir::ArrayAttribute>()
            .AsVector();
    for (size_t i = 0; i < stop_gradients_attr.size(); ++i) {
      auto stop_gradients_attr_j =
          stop_gradients_attr[i].dyn_cast<pir::ArrayAttribute>().AsVector();
      for (size_t j = 0; j < stop_gradients_attr_j.size(); ++j) {
        stop_gradients[0].push_back(
            stop_gradients_attr_j[j].dyn_cast<pir::BoolAttribute>().data());
      }
    }

    VLOG(4) << " op stop_gradients is set ";
  } else {
    std::vector<bool> x_grad_stop_gradient(combine_op_obj_x.inputs().size(),
                                           false);
    stop_gradients[0] = x_grad_stop_gradient;
    VLOG(4) << " op stop_gradients is not set ";
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
  VLOG(4) << "Call Pir Decomposed backward op concat_grad end";
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

std::vector<std::vector<pir::Value>> SliceGradOp::DecompVjp(
    pir::Operation* op) {
  VLOG(4) << "Decomp call slice_grad's decomp interface begin";

  SliceGradOp op_obj = op->dyn_cast<SliceGradOp>();
  (void)op_obj;

  FLAGS_tensor_operants_mode = "static";

  VLOG(6) << "Decomp Prepare inputs of slice_grad";

  Tensor input(std::make_shared<primitive::LazyTensor>(op_obj.input()));
  Tensor out_grad(std::make_shared<primitive::LazyTensor>(op_obj.out_grad()));

  VLOG(6) << "Decomp prepare attributes of slice_grad";
  auto array_list_axes =
      op->attribute("axes").dyn_cast<pir::ArrayAttribute>().AsVector();
  std::vector<int64_t> axes;
  if (array_list_axes.size() > 0) {
    if (array_list_axes[0].isa<pir::Int64Attribute>()) {
      for (size_t i = 0; i < array_list_axes.size(); ++i) {
        axes.push_back(
            array_list_axes[i].dyn_cast<pir::Int64Attribute>().data());
      }

    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "attr is not vector of pir::Int64Attribute "));
    }
  }

  Tensor starts_(std::make_shared<primitive::LazyTensor>(op_obj.starts()));

  auto* starts_define_op =
      std::static_pointer_cast<primitive::LazyTensor>(starts_.impl())
          ->value()
          .defining_op();
  if (starts_define_op->name() != "pd_op.full_int_array") {
    PADDLE_THROW(common::errors::Unimplemented(
        "We don't support dynamic tensors "
        "attribute starts for slice_grad decomposition "
        "for now. "));
  }
  IntArray starts = phi::IntArray(
      paddle::dialect::GetInt64Vector(starts_define_op->attribute("value")));

  Tensor ends_(std::make_shared<primitive::LazyTensor>(op_obj.ends()));

  auto* ends_define_op =
      std::static_pointer_cast<primitive::LazyTensor>(ends_.impl())
          ->value()
          .defining_op();
  if (ends_define_op->name() != "pd_op.full_int_array") {
    PADDLE_THROW(common::errors::Unimplemented(
        "We don't support dynamic tensors "
        "attribute ends for slice_grad decomposition "
        "for now. "));
  }
  IntArray ends = phi::IntArray(
      paddle::dialect::GetInt64Vector(ends_define_op->attribute("value")));

  auto array_list_infer_flags =
      op->attribute("infer_flags").dyn_cast<pir::ArrayAttribute>().AsVector();
  std::vector<int64_t> infer_flags;
  if (array_list_infer_flags.size() > 0) {
    if (array_list_infer_flags[0].isa<pir::Int64Attribute>()) {
      for (size_t i = 0; i < array_list_infer_flags.size(); ++i) {
        infer_flags.push_back(
            array_list_infer_flags[i].dyn_cast<pir::Int64Attribute>().data());
      }

    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "attr is not vector of pir::Int64Attribute "));
    }
  }
  auto array_list_decrease_axis =
      op->attribute("decrease_axis").dyn_cast<pir::ArrayAttribute>().AsVector();
  std::vector<int64_t> decrease_axis;
  if (array_list_decrease_axis.size() > 0) {
    if (array_list_decrease_axis[0].isa<pir::Int64Attribute>()) {
      for (size_t i = 0; i < array_list_decrease_axis.size(); ++i) {
        decrease_axis.push_back(
            array_list_decrease_axis[i].dyn_cast<pir::Int64Attribute>().data());
      }

    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "attr is not vector of pir::Int64Attribute "));
    }
  }

  VLOG(6) << "Decomp call slice_grad's backward composite rule prepare";

  std::vector<std::vector<bool>> stop_gradients(op->results().size());
  if (op->HasAttribute(kAttrStopGradients)) {
    auto stop_gradients_attr = op->attribute(kAttrStopGradients)
                                   .dyn_cast<pir::ArrayAttribute>()
                                   .AsVector();
    stop_gradients[0].push_back(
        stop_gradients_attr[0].dyn_cast<pir::BoolAttribute>().data());
    VLOG(4) << " stop_gradients is set ";
  } else {
    stop_gradients[0].push_back(false);
    VLOG(4) << " stop_gradients is not set ";
  }

  std::vector<std::vector<paddle::Tensor>> tensor_res;
  for (auto arg : stop_gradients) {
    tensor_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "slice_grad";
  FLAGS_tensor_operants_mode = "static";
  VLOG(4) << "Call Pir Decomposed backward op slice_grad";

  paddle::Tensor* input_grad =
      !stop_gradients[0][0] ? &tensor_res[0][0] : nullptr;

  paddle::primitive::details::slice_grad<primitive::LazyTensor>(input,
                                                                out_grad,
                                                                axes,
                                                                starts,
                                                                ends,
                                                                infer_flags,
                                                                decrease_axis,
                                                                input_grad);
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

  VLOG(4) << "Decomp call slice_grad's decomp interface end";
  return res;
}

}  // namespace dialect
}  // namespace paddle
