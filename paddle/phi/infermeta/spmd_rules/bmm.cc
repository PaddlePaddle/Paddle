/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/bmm.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/matmul.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

namespace {

std::vector<int64_t> CheckBmmTensorMeta(const DistMetaTensor& tensor,
                                        const char* tensor_name,
                                        const char* rule_name) {
  const auto shape = common::vectorize(tensor.dims());
  const auto& dims_mapping = tensor.dist_attr().multi_dims_mapping();

  PADDLE_ENFORCE_EQ(shape.size(),
                    3,
                    common::errors::InvalidArgument(
                        "%s expects %s to be a 3-D tensor, but it has rank %d.",
                        rule_name,
                        tensor_name,
                        static_cast<int>(shape.size())));
  PADDLE_ENFORCE_EQ(
      dims_mapping.size(),
      shape.size(),
      common::errors::InvalidArgument(
          "%s expects dims_mapping length of %s (%d) to match its rank (%d).",
          rule_name,
          tensor_name,
          static_cast<int>(dims_mapping.size()),
          static_cast<int>(shape.size())));

  return shape;
}

inline void CheckDimEqual(int64_t lhs,
                          int64_t rhs,
                          const char* lhs_desc,
                          const char* rhs_desc,
                          const char* rule_name) {
  if (lhs != -1 && rhs != -1) {
    PADDLE_ENFORCE_EQ(lhs,
                      rhs,
                      common::errors::InvalidArgument(
                          "%s expects %s (%d) to be equal to %s (%d).",
                          rule_name,
                          lhs_desc,
                          lhs,
                          rhs_desc,
                          rhs));
  }
}

}  // namespace

SpmdInfo BmmInferSpmd(const DistMetaTensor& x, const DistMetaTensor& y) {
  const auto x_shape = CheckBmmTensorMeta(x, "Input(X)", "BmmInferSpmd");
  const auto y_shape = CheckBmmTensorMeta(y, "Input(Y)", "BmmInferSpmd");

  CheckDimEqual(x_shape[2],
                y_shape[1],
                "the last dimension of Input(X)",
                "the second dimension of Input(Y)",
                "BmmInferSpmd");
  CheckDimEqual(x_shape[0],
                y_shape[0],
                "the batch dimension of Input(X)",
                "the batch dimension of Input(Y)",
                "BmmInferSpmd");

  VLOG(6) << "BmmInferSpmd delegates to MatmulInferSpmd (trans_x=false, "
             "trans_y=false).";

  return MatmulInferSpmd(x, y, false, false);
}

SpmdInfo BmmGradInferSpmd(const DistMetaTensor& x,
                          const DistMetaTensor& y,
                          const DistMetaTensor& out_grad) {
  const auto x_shape = CheckBmmTensorMeta(x, "Input(X)", "BmmGradInferSpmd");
  const auto y_shape = CheckBmmTensorMeta(y, "Input(Y)", "BmmGradInferSpmd");
  const auto out_grad_shape =
      CheckBmmTensorMeta(out_grad, "Output@GRAD", "BmmGradInferSpmd");

  CheckDimEqual(x_shape[2],
                y_shape[1],
                "the last dimension of Input(X)",
                "the second dimension of Input(Y)",
                "BmmGradInferSpmd");
  CheckDimEqual(x_shape[0],
                y_shape[0],
                "the batch dimension of Input(X)",
                "the batch dimension of Input(Y)",
                "BmmGradInferSpmd");
  CheckDimEqual(x_shape[0],
                out_grad_shape[0],
                "the batch dimension of Input(X)",
                "the batch dimension of Output@GRAD",
                "BmmGradInferSpmd");
  CheckDimEqual(x_shape[1],
                out_grad_shape[1],
                "the second dimension of Input(X)",
                "the second dimension of Output@GRAD",
                "BmmGradInferSpmd");
  CheckDimEqual(y_shape[2],
                out_grad_shape[2],
                "the last dimension of Input(Y)",
                "the last dimension of Output@GRAD",
                "BmmGradInferSpmd");

  VLOG(6)
      << "BmmGradInferSpmd delegates to MatmulGradInferSpmd (trans_x=false, "
         "trans_y=false).";

  return MatmulGradInferSpmd(x, y, out_grad, false, false);
}
}  // namespace distributed
}  // namespace phi
