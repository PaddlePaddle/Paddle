/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/dropout.h"

#include <vector>

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

// args : (Tensor x, Tensor seed_tensor, Scalar p, bool is_test, str mode, int
// seed, bool fix_seed) output : Tensor(out), Tensor(mask)
SpmdInfo DropoutFwdInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& seed_tensor,
                             Scalar p,
                             bool is_test,
                             const std::string& mode,
                             int seed,
                             bool fix_seed) {
  std::vector<int64_t> seed_tensor_shape =
      common::vectorize(seed_tensor.dims());
  // seed_tensor is None currently
  PADDLE_ENFORCE_EQ(
      IsEmpty(seed_tensor_shape),
      true,
      common::errors::InvalidArgument("seed_tensor should be None"));
  SpmdInfo info = ElementwiseUnaryInferSpmd(x);
  info.first.push_back(seed_tensor.dist_attr());
  info.second.push_back(info.second[0]);
  return info;
}

// args : (Tensor mask, Tensor out_grad, Scalar p, bool is_test, str mode)
// output : Tensor(x_grad)
SpmdInfo DropoutBwdInferSpmd(const DistMetaTensor& mask,
                             const DistMetaTensor& out_grad,
                             Scalar p,
                             bool is_test,
                             const std::string& mode) {
  return ElementwiseBinaryInferSpmd(mask, out_grad);
}
}  // namespace distributed
}  // namespace phi
