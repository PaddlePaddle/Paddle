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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/primitive/rule/vjp/vjp.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/op_base.h"

// TODO(wanghao107)
// this file will be generated in pd_op.cc

namespace paddle {
namespace dialect {
using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<pir::OpResult>> AddNOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  VLOG(6) << "Prepare inputs of add_n_grad";
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      1u,
      platform::errors::InvalidArgument(
          "addn op's inputs size should be 1 but now is %d", inputs_.size()));
  PADDLE_ENFORCE_EQ(
      outputs.size(),
      1u,
      platform::errors::InvalidArgument(
          "addn op's outputs size should be 1 but now is %d", outputs.size()));
  PADDLE_ENFORCE(
      inputs_[0].size() != 0,
      paddle::platform::errors::Fatal("addn op's inputs[0] can't be null"));
  std::vector<Tensor> inputs;
  for (size_t idx = 0; idx < inputs_[0].size(); idx++) {
    inputs.emplace_back(
        std::make_shared<primitive::LazyTensor>(inputs_[0][idx]));
  }

  Tensor out_grad(std::make_shared<primitive::LazyTensor>(out_grads[0][0]));

  VLOG(6) << "Vjp prepare Prepare attributes of add_n_grad";

  VLOG(6) << "Vjp prepare call add_n's vjp inteface";

  std::vector<std::vector<Tensor>> tensor_res =
      primitive::add_n_vjp(inputs, out_grad, stop_gradients);

  VLOG(6) << "Vjp prepare stop gradient of add_n_grad";

  std::vector<std::vector<pir::OpResult>> res(tensor_res.size());
  for (size_t i = 0; i < tensor_res.size(); ++i) {
    res[i].resize(tensor_res[i].size());
    for (size_t j = 0; j < tensor_res[i].size(); ++j) {
      if (tensor_res[i][j].defined()) {
        res[i][j] = std::static_pointer_cast<primitive::LazyTensor>(
                        tensor_res[i][j].impl())
                        ->value()
                        .dyn_cast<pir::OpResult>();
      }
    }
  }
  return res;
}

std::vector<std::vector<pir::OpResult>> ExpandOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  PADDLE_ENFORCE_EQ(inputs_.size(),
                    2,
                    platform::errors::InvalidArgument(
                        "expand op's inputs size should be 2, but now is %d.",
                        inputs_.size()));
  PADDLE_ENFORCE_EQ(outputs.size(),
                    1,
                    platform::errors::InvalidArgument(
                        "expand op's outputs size should be 1, but now is %d.",
                        outputs.size()));

  VLOG(6) << "Prepare inputs of expand_grad";

  Tensor x(std::make_shared<primitive::LazyTensor>(inputs_[0][0]));
  Tensor out_grad(std::make_shared<primitive::LazyTensor>(out_grads[0][0]));

  VLOG(6) << "Vjp prepare Prepare attributes of expand_grad";

  Tensor shape(std::make_shared<primitive::LazyTensor>(inputs_[1][0]));

  VLOG(6) << "Vjp prepare call expand's vjp inteface";

  std::vector<std::vector<Tensor>> tensor_res =
      primitive::expand_vjp(x, out_grad, shape, stop_gradients);

  VLOG(6) << "Vjp prepare stop gradient of expand_grad";

  std::vector<std::vector<pir::OpResult>> res(tensor_res.size());
  for (size_t i = 0; i < tensor_res.size(); ++i) {
    res[i].resize(tensor_res[i].size());
    for (size_t j = 0; j < tensor_res[i].size(); ++j) {
      if (tensor_res[i][j].defined()) {
        res[i][j] = std::static_pointer_cast<primitive::LazyTensor>(
                        tensor_res[i][j].impl())
                        ->value()
                        .dyn_cast<pir::OpResult>();
      }
    }
  }
  return res;
}

std::vector<std::vector<pir::OpResult>> IncrementOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      1,
      platform::errors::InvalidArgument(
          "Increment op's inputs size should be 2, but now is %d.",
          inputs_.size()));
  PADDLE_ENFORCE_EQ(
      outputs.size(),
      1,
      platform::errors::InvalidArgument(
          "Increment op's outputs size should be 1, but now is %d.",
          outputs.size()));

  VLOG(6) << "Vjp prepare Prepare attributes of increment_grad";

  float value = op->attribute("value").dyn_cast<pir::FloatAttribute>().data();

  VLOG(6) << "Vjp prepare call increment's vjp inteface";

  pir::OpResult tensor_res = paddle::dialect::increment(inputs_[0][0], -value);

  std::vector<std::vector<pir::OpResult>> res{{tensor_res}};

  return res;
}

std::vector<std::vector<pir::OpResult>> Increment_Op::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      1,
      platform::errors::InvalidArgument(
          "Increment_ op's inputs size should be 1, but now is %d.",
          inputs_.size()));
  PADDLE_ENFORCE_EQ(
      outputs.size(),
      1,
      platform::errors::InvalidArgument(
          "Increment_ op's outputs size should be 1, but now is %d.",
          outputs.size()));

  VLOG(6) << "Vjp prepare Prepare attributes of increment__grad";

  float value = op->attribute("value").dyn_cast<pir::FloatAttribute>().data();

  VLOG(6) << "Vjp prepare call increment_'s vjp inteface";

  paddle::dialect::increment_(inputs_[0][0], -value);

  std::vector<std::vector<pir::OpResult>> res;
  return res;
}

std::vector<std::vector<pir::OpResult>> ArrayWrite_Op::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& in_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      3,
      platform::errors::InvalidArgument(
          "ArrayWrite_ op's inputs size should be 3, but now is %d.",
          inputs_.size()));
  PADDLE_ENFORCE_EQ(
      outputs.size(),
      1,
      platform::errors::InvalidArgument(
          "ArrayWrite_ op's outputs size should be 1, but now is %d.",
          outputs.size()));

  PADDLE_ENFORCE_EQ(
      in_grads.size(),
      1,
      platform::errors::InvalidArgument(
          "ArrayWrite_ op's outputs size should be 1, but now is %d.",
          outputs.size()));

  VLOG(6) << "Vjp prepare call  ArrayWrite_'s vjp inteface";
  pir::OpResult x_grad =
      paddle::dialect::array_read(in_grads[0][0], inputs_[2][0]);
  pir::OpResult zero = paddle::dialect::zeros_like(inputs_[1][0]);
  paddle::dialect::array_write_(in_grads[0][0], zero, inputs_[2][0]);
  std::vector<std::vector<pir::OpResult>> res(1);
  res[0].resize(1);
  if (!stop_gradients[0][0]) {
    res[0][0] = x_grad;
  }
  return res;
}

std::vector<std::vector<pir::OpResult>> ArrayReadOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      2,
      platform::errors::InvalidArgument(
          "Array_read op's inputs size should be 2, but now is %d.",
          inputs_.size()));
  PADDLE_ENFORCE_EQ(
      outputs.size(),
      1,
      platform::errors::InvalidArgument(
          "Array_read op's outputs size should be 1, but now is %d.",
          outputs.size()));
  // x = array_read(input, i)
  // out_grads[0][0] is x_grad
  // out_grads[1][0] is input_array_grad
  PADDLE_ENFORCE_EQ(
      out_grads.size(),
      2,
      platform::errors::InvalidArgument(
          "Array_read op's outputs size should be 1, but now is %d.",
          out_grads.size()));

  VLOG(6) << "Vjp prepare call  Array_read's vjp inteface";

  pir::Value array_grad_i_origin =
      paddle::dialect::array_read(out_grads[1][0], inputs_[1][0]);
  pir::Value array_grad_i =
      paddle::dialect::add(array_grad_i_origin, out_grads[0][0]);
  paddle::dialect::array_write_(out_grads[1][0], array_grad_i, inputs_[1][0]);

  std::vector<std::vector<pir::OpResult>> res;
  return res;
}

std::vector<std::vector<pir::OpResult>> ArrayToTensorOp::Vjp(
    pir::Operation* op,
    const std::vector<std::vector<pir::Value>>& inputs_,
    const std::vector<std::vector<pir::Value>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  PADDLE_ENFORCE_EQ(
      inputs_.size(),
      1,
      platform::errors::InvalidArgument(
          "Array_read op's inputs size should be 1, but now is %d.",
          inputs_.size()));
  PADDLE_ENFORCE_EQ(
      outputs.size(),
      2,
      platform::errors::InvalidArgument(
          "Array_read op's outputs size should be 2, but now is %d.",
          outputs.size()));

  PADDLE_ENFORCE_EQ(
      out_grads.size(),
      2,
      platform::errors::InvalidArgument(
          "Array_read op's outputs size should be 2, but now is %d.",
          out_grads.size()));

  VLOG(6) << "Vjp prepare Prepare attributes of array_to_tensor_grad";
  int axis = op->attribute("axis").dyn_cast<pir::Int32Attribute>().data();
  bool use_stack =
      op->attribute("use_stack").dyn_cast<pir::BoolAttribute>().data();

  VLOG(6) << "Vjp prepare call ArrayToTensor's vjp inteface";

  pir::OpResult tensor_res = paddle::dialect::tensor_to_array(
      inputs_[0][0], out_grads[0][0], axis, use_stack);

  std::vector<std::vector<pir::OpResult>> res(1);
  res[0].resize(1);
  if (!stop_gradients[0][0]) {
    res[0][0] = tensor_res;
  }
  return res;
}

}  // namespace dialect
}  // namespace paddle
