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
#ifdef GET_OP_LIST
#undef GET_OP_LIST
paddle::dialect::PyLayerOp
#else

#include <unordered_map>

#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/shape_optimization_pass.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_type.h"

namespace paddle {
namespace dialect {

namespace py = ::pybind11;

static std::unordered_map<uint64_t, py::object> g_backward_py_callables;

// Return pybind11::object* instead of pybind11::object
// Returning pybind11::object would cause reference count increasing
// but without GIL, reference count in Python may not be safe
static py::object *GetPythonCallableObject(uint64_t unique_id) {
  PADDLE_ENFORCE_NE(
      g_backward_py_callables.find(unique_id),
      g_backward_py_callables.end(),
      platform::errors::InvalidArgument(
          "Unique_id %d is not found in g_backward_py_callables. The possible "
          "reasons are below:"
          "1. The callable function was not registered for `unique_id` by "
          "`RegisterPyCallableObject`"
          "2. The callable function was remove from g_backward_py_callables",
          unique_id));
  return &(g_backward_py_callables[unique_id]);
}

// NOTE: Use to manage the context of pylayer op constructing block
class PyLayerBlockContextManager {
 public:
  explicit PyLayerBlockContextManager(pir::Block *block) {
    ApiBuilder::Instance().PushInsertionPoint();
    ApiBuilder::Instance().SetInsertionPointToBlockEnd(block);
  }

  ~PyLayerBlockContextManager() { ApiBuilder::Instance().LoadInsertionPoint(); }

  PyLayerBlockContextManager(const PyLayerBlockContextManager &) = delete;
  PyLayerBlockContextManager &operator=(const PyLayerBlockContextManager &) =
      delete;

 private:
  // disable default constructor
  PyLayerBlockContextManager() {}
};

void CallPythonFunc(py::object *callable,
                    const std::vector<pir::Value> &ins,
                    std::vector<pir::Value> *outs) {
  py::gil_scoped_acquire guard;
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    in_args[i] = py::cast(ins[i]);
  }

  auto ret = (*callable)(*in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  size_t ret_num = py::len(ret_tuple);
  size_t out_num = outs->size();
  if (UNLIKELY(ret_num != out_num)) {
    // Python function has no return values or returns None
    // In this case, ret_num = 1 && ret[0] == None && out_num should be 0
    // Otherwise, ret_num must be equal to out_num
    PADDLE_ENFORCE_EQ(ret_num == 1,
                      true,
                      platform::errors::InvalidArgument(
                          "Python function has no return values or returns "
                          "None. In this case, ret_num = 1 && ret[0] == None "
                          "&& out_num should be 0. But ret_num is %d",
                          ret_num));

    PADDLE_ENFORCE_EQ(
        out_num == 0,
        true,
        platform::errors::InvalidArgument(
            "Python function has no return values or returns None. In "
            "this case, ret_num = 1 && ret[0] == None && out_num should "
            "be 0. But out_num is %d",
            out_num));

    PADDLE_ENFORCE_EQ(
        py::cast<pir::Value *>(ret_tuple[0]) == nullptr,
        true,
        platform::errors::InvalidArgument(
            "Python function has no return values or returns None. In "
            "this case, ret_num = 1 && ret[0] == None && out_num should "
            "be 0. But ret[0] is not None"));
  }

  for (size_t i = 0; i < out_num; ++i) {
    try {
      // NOTE(MarioLulab): why can cast ? Might release Value in dangerous.
      auto py_out_value = py::cast<pir::Value>(ret_tuple[i]);
      PADDLE_ENFORCE_NOT_NULL(py_out_value.impl(),
                              platform::errors::InvalidArgument(
                                  "Output value %d should not be nullptr", i));
      (*outs)[i] = py_out_value;
    } catch (py::cast_error &) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "pybind11::cast to pir::Value error. The %d-th output exception is "
          "pir::Value",
          i));
    }
  }
}

void PyLayerOp::Build(pir::Builder &builder,             // NOLINT
                      pir::OperationArgument &argument,  // NOLINT
                      pir::Value combined_inputs,
                      std::vector<pir::Type> &&output_types) {
  argument.AddInput(combined_inputs);
  argument.output_types.swap(output_types);
  argument.AddRegion().emplace_back();
}

void PyLayerOp::Build(pir::Builder &builder,             // NOLINT
                      pir::OperationArgument &argument,  // NOLINT
                      pir::Value combined_inputs,
                      std::unique_ptr<pir::Block> &&fwd_block) {
  VLOG(4) << "Start build PyLayerOp";
  if (fwd_block && !fwd_block->empty() &&
      fwd_block->back().isa<pir::YieldOp>()) {
    auto &op = fwd_block->back();

    std::vector<pir::Attribute> outs_stop_gradient;
    for (size_t i = 0; i < op.num_operands(); ++i) {
      argument.AddOutput(op.operand(i).type());
      auto bool_attr = op.operand_source(i).attribute<pir::BoolAttribute>(
          pir::kStopGradientAttrName);
      outs_stop_gradient.push_back(bool_attr ? bool_attr
                                             : builder.bool_attr(false));
    }

    argument.AddAttribute(
        pir::kStopGradientAttrName,
        pir::ArrayAttribute::get(builder.ir_context(), outs_stop_gradient));
  }

  argument.AddRegion().push_back(fwd_block.release());
  argument.AddInput(combined_inputs);
}

pir::Block &PyLayerOp::forward_block() {
  pir::Region &region = forward_region();
  if (region.empty()) {
    region.emplace_back();
  }

  return region.front();
}

void PyLayerOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " = pd_op.pylayer";
  printer.PrintOpOperands(op);
  os << " -> ";
  printer.PrintOpReturnType(op);
  os << "{";
  for (auto &item : forward_block()) {
    os << "\n  ";
    printer.PrintOperation(&item);
  }
  os << "\n }";
}

void PyLayerOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: PyLayerOp.";
  // NOTE(MarioLulab): do nothing.
}

void PyLayerOp::VerifyRegion() {
  VLOG(4) << "Start Verifying sub regions for: PyLayerOp.";
  VLOG(4) << "Start Verifying forward block.";
  PADDLE_ENFORCE_EQ((*this)->region(0).size(),
                    1u,
                    phi::errors::PreconditionNotMet(
                        "The size %d of forward_region must be 1.",
                        (*this)->region(0).size()));
  if ((*this)->num_results() != 0) {
    auto &fwd_last_op = (*this)->region(0).front().back();
    PADDLE_ENFORCE_EQ(true,
                      fwd_last_op.isa<pir::YieldOp>(),
                      phi::errors::PreconditionNotMet(
                          "The last of forward block must be YieldOp"));
    PADDLE_ENFORCE_EQ(
        fwd_last_op.num_operands(),
        (*this)->num_results(),
        phi::errors::PreconditionNotMet(
            "The size of last of forward block op's input must be "
            "equal to PyLayerOp's outputs num."));
  }
}

void PyLayerOp::UpdateOutput() {
  PADDLE_ENFORCE_NOT_NULL(*this,
                          paddle::platform::errors::InvalidArgument(
                              "The pylayer_op in PyLayerOp used to update "
                              "output can't be nullptr"));
  auto block = parent();
  PADDLE_ENFORCE_NOT_NULL(
      block,
      paddle::platform::errors::InvalidArgument(
          "The parent block of pylayer_op which used to update "
          "output can't be nullptr"));
  pir::Block::Iterator iter = **this;
  pir::Builder builder(ir_context(), false);
  auto new_pylayer_op =
      builder.Build<PyLayerOp>(combined_inputs(), forward_region().TakeBack());
  block->Assign(iter, new_pylayer_op);
  PyLayerOp::operator=(new_pylayer_op);
  VerifyRegion();
}

std::vector<std::vector<pir::Value>> PyLayerOp::Vjp(
    pir::Operation *op,
    const std::vector<std::vector<pir::Value>> &inputs_,
    const std::vector<std::vector<pir::Value>> &outputs,
    const std::vector<std::vector<pir::Value>> &out_grads,
    const std::vector<std::vector<bool>> &stop_gradients) {
  std::vector<pir::Type> output_types;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (!stop_gradients[i][0]) {
      output_types.push_back(inputs_[i][0].type());
    }
  }

  std::vector<pir::Value> output_grads;
  for (size_t i = 0; i < out_grads.size(); ++i) {
    output_grads.push_back(out_grads[i][0]);
  }

  auto out_grads_combine_op =
      ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>(output_grads);

  auto pylayer_grad = ApiBuilder::Instance().GetBuilder()->Build<PyLayerOp>(
      out_grads_combine_op.out(), std::move(output_types));

  std::vector<pir::Value> pylayer_grad_inputs(output_types.size());
  auto *py_callable = GetPythonCallableObject(op->id());

  {
    // enter block of pylayer_grad
    PyLayerBlockContextManager(&(pylayer_grad.forward_block()));
    CallPythonFunc(py_callable, output_grads, &pylayer_grad_inputs);

    // append yield op for outputs value
    ApiBuilder::Instance().GetBuilder()->Build<pir::YieldOp>(
        pylayer_grad_inputs);

    // exit block of pylayer_grad
  }

  std::vector<std::vector<pir::Value>> res{inputs_.size()};
  for (size_t i = 0; i < output_types.size(); ++i) {
    res[i].resize(1);
    res[i][0] = pylayer_grad->result(i);
  }
  return res;
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PyLayerOp)

#endif
