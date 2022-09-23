// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/lib/api_custom_impl.h"
DECLARE_bool(check_nan_inf);

paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                     egr::kSlotSmallVectorSize>
AddNGradNodeFinal::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  // Fill Zero For GradIn Tensors

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto x = egr::EagerUtils::RecoverTensorWrapper(&this->x_);
  auto& out_grad = hooked_grads[0][0];
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       egr::kSlotSmallVectorSize>
      returns(1);
  for (int i = 0; i < 1; ++i) {
    out_metas[i].size() == 0 ? returns[i].resize(1)
                             : returns[i].resize(out_metas[i].size());
  }

  std::vector<paddle::experimental::Tensor*> api_output_0;
  api_output_0.reserve(returns[0].size());
  for (size_t i = 0; i < returns[0].size(); ++i) {
    if (out_metas[0].empty() || out_metas[0][i].IsStopGradient()) {
      api_output_0.push_back(nullptr);
    } else {
      api_output_0.push_back(&returns[0][i]);
    }
  }
  // Call grad_api function
  VLOG(3) << "Final State Running: AddNGradNodeFinal";

  // dygraph function
  for (size_t i = 0; i < returns[0].size(); i++) {
    returns[0][i] = ::scale_ad_func(out_grad, phi::Scalar(1.0), 0.0, true);
  }

  // Check NaN and Inf id needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("add_n_grad", returns);
  }

  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}
