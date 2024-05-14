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

std::vector<std::vector<pir::Value>> AddGradOp::DecompVjp(pir::Operation* op) {
  VLOG(4) << "Decomp call add_grad's decomp interface begin";

  AddGradOp op_obj = op->dyn_cast<AddGradOp>();
  (void)op_obj;

  FLAGS_tensor_operants_mode = "static";

  VLOG(6) << "Decomp Prepare inputs of add_grad";

  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));
  Tensor y(std::make_shared<primitive::LazyTensor>(op_obj.y()));
  Tensor out_grad(std::make_shared<primitive::LazyTensor>(op_obj.out_grad()));

  VLOG(6) << "Decomp prepare attributes of add_grad";
  int axis = op->attribute("axis").dyn_cast<pir::Int32Attribute>().data();

  VLOG(6) << "Decomp call add_grad's composite rule prepare";

  std::vector<std::vector<bool>> stop_gradients(op->results().size());
  stop_gradients[0].push_back(false);
  stop_gradients[1].push_back(false);

  std::vector<std::vector<paddle::Tensor>> tensor_res;
  for (auto arg : stop_gradients) {
    tensor_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "add_grad";
  FLAGS_tensor_operants_mode = "static";
  VLOG(4) << "Call Pir Decomposed backward op add_grad";
  paddle::Tensor* x_grad = !stop_gradients[0][0] ? &tensor_res[0][0] : nullptr;
  paddle::Tensor* y_grad = !stop_gradients[1][0] ? &tensor_res[1][0] : nullptr;
  paddle::primitive::details::add_grad<primitive::LazyTensor>(
      x, y, out_grad, axis, x_grad, y_grad);
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
  return res;
}

}  // namespace dialect
}  // namespace paddle
