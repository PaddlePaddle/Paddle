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

#include <sstream>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"

PD_DECLARE_KERNEL(tanh, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(pow, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply, CPU, ALL_LAYOUT);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(tanh, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(tanh_grad, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(pow, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply, CPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace prim {

TEST(EagerPrim, TanhBackwardTest) {
  // 1. Initialized
  eager_test::InitEnv(paddle::platform::CPUPlace());
  // 2. pre
  paddle::framework::DDim ddim = phi::make_ddim({4, 16, 16, 32});
  paddle::experimental::Tensor tensor =
      egr_utils_api::CreateTensorWithValue(ddim,
                                           paddle::platform::CPUPlace(),
                                           phi::DataType::FLOAT32,
                                           phi::DataLayout::NCHW,
                                           5.0 /*value*/,
                                           true /*is_leaf*/);
  egr_utils_api::RetainGradForTensor(tensor);

  // 3. Run Forward
  paddle::experimental::Tensor out = tanh_ad_func(
      tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/);

  std::vector<paddle::experimental::Tensor> outs = {out};

  // 4. Run Backward
  Backward(outs, {});

  VLOG(7) << "Target Grad is: "
          << std::static_pointer_cast<phi::DenseTensor>(
                 EagerUtils::unsafe_autograd_meta(tensor)->Grad().impl())
                 ->data<float>()[0];
  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 2.0);
}

}  // namespace prim
}  // namespace paddle
