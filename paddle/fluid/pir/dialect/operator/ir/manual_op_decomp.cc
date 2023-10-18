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

std::vector<std::vector<pir::OpResult>> MeanOp::Decomp(pir::Operation* op) {
  MeanOp op_obj = op->dyn_cast<MeanOp>();
  (void)op_obj;

  VLOG(4) << "Decomp Prepare inputs of mean";

  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));

  VLOG(4) << "Decomp prepare attributes of mean";

  IntArray axis = op->attribute("axis")
                      .dyn_cast<paddle::dialect::IntArrayAttribute>()
                      .data();

  bool keepdim = op->attribute("keepdim").dyn_cast<pir::BoolAttribute>().data();
  VLOG(4) << "Decomp mean keep_dim " << keepdim;

  VLOG(4) << "Decomp prepare call mean's decomp interface";

  Tensor op_res =
      paddle::primitive::details::mean_decomp<primitive::LazyTensor>(
          x, axis, keepdim);

  auto org_res = op->results();
  std::vector<std::vector<pir::OpResult>> res(org_res.size());
  res[0].push_back(
      std::static_pointer_cast<primitive::LazyTensor>(op_res.impl())
          ->value()
          .dyn_cast<pir::OpResult>());
  return res;
}

}  // namespace dialect
}  // namespace paddle
