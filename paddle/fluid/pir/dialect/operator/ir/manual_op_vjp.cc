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
    const std::vector<std::vector<pir::OpResult>>& outputs,
    const std::vector<std::vector<pir::Value>>& out_grads,
    const std::vector<std::vector<bool>>& stop_gradients) {
  AddNOp op_obj = op->dyn_cast<AddNOp>();

  VLOG(6) << "Prepare inputs of add_n_grad";
  PADDLE_ENFORCE(
      op_obj.inputs() != nullptr,
      paddle::platform::errors::Fatal("addn op's inputs can't be null"));
  pir::CombineOp combine_op_obj = op_obj.inputs()
                                      .dyn_cast<pir::OpResult>()
                                      .owner()
                                      ->dyn_cast<pir::CombineOp>();
  std::vector<Tensor> inputs;
  for (size_t idx = 0; idx < combine_op_obj.inputs().size(); idx++) {
    inputs.emplace_back(
        std::make_shared<primitive::LazyTensor>(combine_op_obj.inputs()[idx]));
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

}  // namespace dialect
}  // namespace paddle
