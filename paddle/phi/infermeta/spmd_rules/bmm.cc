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
                                        const char* rank_error_msg,
                                        const char* dims_mapping_error_msg) {
  const auto shape = common::vectorize(tensor.dims());
  const auto& dims_mapping = tensor.dist_attr().multi_dims_mapping();

  PADDLE_ENFORCE_EQ(shape.size(),
                    3,
                    common::errors::InvalidArgument(
                        rank_error_msg, static_cast<int>(shape.size())));
  PADDLE_ENFORCE_EQ(
      dims_mapping.size(),
      shape.size(),
      common::errors::InvalidArgument(dims_mapping_error_msg,
                                      static_cast<int>(dims_mapping.size()),
                                      static_cast<int>(shape.size())));

  return shape;
}

inline void CheckDimEqual(int64_t lhs, int64_t rhs, const char* msg) {
  if (lhs != -1 && rhs != -1) {
    PADDLE_ENFORCE_EQ(lhs, rhs, common::errors::InvalidArgument(msg, lhs, rhs));
  }
}

}  // namespace

SpmdInfo BmmInferSpmd(const DistMetaTensor& x, const DistMetaTensor& y) {
  const auto x_shape = CheckBmmTensorMeta(
      x,
      "BmmInferSpmd requires Input(X) to be a 3-D tensor, but got rank [%d].",
      "BmmInferSpmd expects Input(X)'s dims_mapping size [%d] to match its "
      "rank [%d].");
  const auto y_shape = CheckBmmTensorMeta(
      y,
      "BmmInferSpmd requires Input(Y) to be a 3-D tensor, but got rank [%d].",
      "BmmInferSpmd expects Input(Y)'s dims_mapping size [%d] to match its "
      "rank [%d].");

  CheckDimEqual(x_shape[2],
                y_shape[1],
                "BmmInferSpmd expects Input(X)'s width [%d] to equal "
                "Input(Y)'s height [%d].");
  CheckDimEqual(x_shape[0],
                y_shape[0],
                "BmmInferSpmd expects Input(X) and Input(Y) to share the "
                "same batch size [%d] vs [%d].");

  VLOG(6) << "BmmInferSpmd delegates to MatmulInferSpmd (trans_x=false, "
             "trans_y=false).";

  return MatmulInferSpmd(x, y, false, false);
}

SpmdInfo BmmGradInferSpmd(const DistMetaTensor& x,
                          const DistMetaTensor& y,
                          const DistMetaTensor& out_grad) {
  const auto x_shape =
      CheckBmmTensorMeta(x,
                         "BmmGradInferSpmd requires Input(X) to be a 3-D "
                         "tensor, but got rank [%d].",
                         "BmmGradInferSpmd expects Input(X)'s dims_mapping "
                         "size [%d] to match its rank [%d].");
  const auto y_shape =
      CheckBmmTensorMeta(y,
                         "BmmGradInferSpmd requires Input(Y) to be a 3-D "
                         "tensor, but got rank [%d].",
                         "BmmGradInferSpmd expects Input(Y)'s dims_mapping "
                         "size [%d] to match its rank [%d].");
  const auto out_grad_shape =
      CheckBmmTensorMeta(out_grad,
                         "BmmGradInferSpmd requires Output@Grad to be a 3-D "
                         "tensor, but got rank [%d].",
                         "BmmGradInferSpmd expects Output@Grad's dims_mapping "
                         "size [%d] to match its rank [%d].");

  CheckDimEqual(x_shape[2],
                y_shape[1],
                "BmmGradInferSpmd expects Input(X)'s width [%d] to equal "
                "Input(Y)'s height [%d].");
  CheckDimEqual(x_shape[0],
                y_shape[0],
                "BmmGradInferSpmd expects Input(X) and Input(Y) to share the "
                "same batch size [%d] vs [%d].");
  CheckDimEqual(x_shape[0],
                out_grad_shape[0],
                "BmmGradInferSpmd expects Output@Grad's batch size [%d] to "
                "match Input(X)'s [%d].");
  CheckDimEqual(x_shape[1],
                out_grad_shape[1],
                "BmmGradInferSpmd expects Output@Grad's second dimension "
                "[%d] to match Input(X)'s second dimension [%d].");
  CheckDimEqual(y_shape[2],
                out_grad_shape[2],
                "BmmGradInferSpmd expects Output@Grad's third dimension [%d] "
                "to match Input(Y)'s third dimension [%d].");

  VLOG(6)
      << "BmmGradInferSpmd delegates to MatmulGradInferSpmd (trans_x=false, "
         "trans_y=false).";

  return MatmulGradInferSpmd(x, y, out_grad, false, false);
}
}  // namespace distributed
}  // namespace phi
