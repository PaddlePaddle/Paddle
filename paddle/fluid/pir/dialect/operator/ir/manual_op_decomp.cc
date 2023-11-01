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
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/op_base.h"

// TODO(chenzhuo)
// this file will be generated in pd_op_decomp.cc

namespace paddle {
namespace dialect {
using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<pir::OpResult>> SqueezeOp::Decomp(pir::Operation* op) {
  SqueezeOp op_obj = op->dyn_cast<SqueezeOp>();
  (void)op_obj;

  VLOG(4) << "Decomp Prepare inputs of squeeze";

  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));
  Tensor axis_(std::make_shared<primitive::LazyTensor>(op_obj.axis()));

  VLOG(4) << "Decomp prepare attributes of squeeze";

  auto* axis_define_op =
      std::static_pointer_cast<primitive::LazyTensor>(axis_.impl())
          ->value()
          .dyn_cast<pir::OpResult>()
          .owner();
  if (axis_define_op->name() != "pd_op.full_int_array") {
    PADDLE_THROW(
        platform::errors::Unimplemented("We don't support dynamic tensors "
                                        "attribute axis for max_grad composite "
                                        "for now. "));
  }
  IntArray axis = phi::IntArray(
      paddle::dialect::GetInt64Vector(axis_define_op->attribute("value")));

  //   IntArray axis =
  //   op->attribute("axis").dyn_cast<paddle::dialect::IntArrayAttribute>().data();

  VLOG(4) << "Decomp prepare call squeeze's decomp interface";

  auto org_res = op->results();
  std::vector<std::vector<pir::OpResult>> res(org_res.size());

  std::tuple<Tensor, Tensor> op_res =
      paddle::primitive::details::squeeze_decomp<primitive::LazyTensor>(x,
                                                                        axis);
  //   for (size_t i = 0; i < 2; i++) {
  //       res[i].push_back(std::static_pointer_cast<primitive::LazyTensor>(std::get<i>(op_res).impl())->value().dyn_cast<pir::OpResult>());
  //       }

  res[0].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                       std::get<0>(op_res).impl())
                       ->value()
                       .dyn_cast<pir::OpResult>());
  VLOG(4) << "Finish Decomp call squeeze's decomp interface out 0";
  pir::OpResult temp;

  res[1].push_back(temp);
  VLOG(4) << "Finish Decomp call squeeze's decomp interface out 1";

  return res;
}

}  // namespace dialect
}  // namespace paddle
