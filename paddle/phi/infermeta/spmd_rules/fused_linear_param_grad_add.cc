/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/fused_linear_param_grad_add.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/matmul.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo FusedLinearParamGradAddInferSpmd(const DistMetaTensor& x,
                                          const DistMetaTensor& dout,
                                          const DistMetaTensor& dweight,
                                          const DistMetaTensor& dbias,
                                          bool multi_precision,
                                          bool has_bias) {
  auto dy_spmd_info =
      MatmulInferSpmd(x, dout, /*trans_x=*/true, /*trans_y=*/false);
  auto& x_dist_attr = PADDLE_GET_CONST(TensorDistAttr, dy_spmd_info.first[0]);
  auto& dout_dist_attr =
      PADDLE_GET_CONST(TensorDistAttr, dy_spmd_info.first[1]);
  auto weight_grad_dist_attr =
      PADDLE_GET_CONST(TensorDistAttr, dy_spmd_info.second[0]);

  weight_grad_dist_attr = ReduceGradBroadCastDims(2, weight_grad_dist_attr);

  TensorDistAttr dweight_dist_attr = dweight.dist_attr();
  auto dweight_shape = common::vectorize(dweight.dims());
  TensorDistAttr dbias_dist_attr = dbias.dist_attr();
  auto dbias_shape = common::vectorize(dbias.dims());

  TensorDistAttr bias_grad_dist_attr;
  if (has_bias) {
    bias_grad_dist_attr = ReduceGradBroadCastDims(1, dout.dist_attr());
  }

  // check dweight and dweight_grad
  if (!IsEmpty(dweight_shape)) {
    PADDLE_ENFORCE_EQ(dweight_dist_attr,
                      weight_grad_dist_attr,
                      common::errors::InvalidArgument(
                          "dweight_dist_attr [%s] and weight_grad_dist_attr "
                          "[%s] should be equal",
                          dweight_dist_attr.to_string(),
                          weight_grad_dist_attr.to_string()));
  }
  // check dbias and bias_grad
  if (!IsEmpty(dbias_shape)) {
    PADDLE_ENFORCE_EQ(
        dbias_dist_attr,
        bias_grad_dist_attr,
        common::errors::InvalidArgument(
            "dbias_dist_attr [%s] and bias_grad_dist_attr [%s] should be equal",
            dbias_dist_attr.to_string(),
            bias_grad_dist_attr.to_string()));
  }

  return {{x_dist_attr, dout_dist_attr, dweight_dist_attr, dbias_dist_attr},
          {weight_grad_dist_attr, bias_grad_dist_attr}};
}

SpmdInfo FusedLinearParamGradAddInferSpmdFakeReverse() { return SpmdInfo(); }

}  // namespace phi::distributed
