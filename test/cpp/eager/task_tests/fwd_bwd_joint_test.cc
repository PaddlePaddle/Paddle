// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/scale_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "test/cpp/eager/test_utils.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(full, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, KPS, ALL_LAYOUT);
#endif

namespace egr {

paddle::Tensor hook_function(const paddle::Tensor& t) {
  auto t_dense = std::dynamic_pointer_cast<phi::DenseTensor>(t.impl());

  auto ret_meta = phi::DenseTensorMeta(
      t_dense->dtype(), t_dense->dims(), t_dense->layout());
  auto place = t_dense->place();
  size_t bytes_size =
      common::product(t_dense->dims()) * SizeOf(t_dense->dtype());
  auto ret_dense = std::make_shared<phi::DenseTensor>(
      paddle::memory::Alloc(place, bytes_size), std::move(ret_meta));

  float* t_ptr = t_dense->mutable_data<float>(place);
  float* ret_ptr = ret_dense->mutable_data<float>(place);
  for (int i = 0; i < ret_dense->numel(); i++) {
    ret_ptr[i] = t_ptr[i] + 5.0f;
  }

  auto ret_impl = std::dynamic_pointer_cast<phi::TensorBase>(ret_dense);
  paddle::Tensor ret = paddle::Tensor();
  ret.set_impl(ret_impl);

  return ret;
}

TEST(FwdBwdJoint, SingleNode) {
  eager_test::InitEnv(phi::CPUPlace());

  // 1. Prepare Input
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  egr_utils_api::RetainGradForTensor(tensor);

  // 3. Run Forward
  float scale = 2.0;
  float bias = 3.0;
  paddle::Tensor out = egr::scale(
      tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output
  eager_test::CompareTensorWithValue<float>(out, 13.0);

  std::vector<paddle::Tensor> outs = {out};
  // 4. Run Backward
  Backward(outs, {});

  VLOG(7) << "Target Grad is: "
          << std::static_pointer_cast<phi::DenseTensor>(
                 EagerUtils::unsafe_autograd_meta(tensor)->Grad().impl())
                 ->data<float>()[0];
  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 2.0);
}

/*
 inp
  |
Node0
  |
Node1
  |
 out
*/
TEST(FwdBwdJoint, LinearNodes) {
  eager_test::InitEnv(phi::CPUPlace());

  // 1. Prepare Input
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  egr_utils_api::RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  paddle::Tensor out0 = egr::scale(tensor,
                                   scale0,
                                   bias0,
                                   true /*bias_after_scale*/,
                                   true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  paddle::Tensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output 0
  eager_test::CompareTensorWithValue<float>(out0, 13.0);

  // Examine Forward Output 1
  eager_test::CompareTensorWithValue<float>(out1, 75.0);

  std::vector<paddle::Tensor> outs = {out1};
  // 4. Run Backward
  Backward(outs, {});

  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 10.0);
}

/*
       inp
        |
      Node0
    ____|____
    |       |
  Node1   Node2
    |       |
   out1    out2
*/
TEST(FwdBwdJoint, BranchedNodes) {
  eager_test::InitEnv(phi::CPUPlace());

  // 1. Prepare Input
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  egr_utils_api::RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  paddle::Tensor out0 = egr::scale(tensor,
                                   scale0,
                                   bias0,
                                   true /*bias_after_scale*/,
                                   true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  paddle::Tensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  paddle::Tensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output 0
  eager_test::CompareTensorWithValue<float>(out0, 13.0);

  // Examine Forward Output 1
  eager_test::CompareTensorWithValue<float>(out1, 75.0);

  // Examine Forward Output 2
  {
    auto dense_out = std::dynamic_pointer_cast<phi::DenseTensor>(out2.impl());
    float* ptr = dense_out->mutable_data<float>(phi::CPUPlace());
    for (int i = 0; i < 20; i++) {
      PADDLE_ENFORCE(ptr[i] == 150.0,
                     common::errors::Fatal(
                         "Detected numerical Error, Expected %f but got %f",
                         150.0,
                         ptr[i]));
    }
  }

  // 4. Run Backward
  std::vector<paddle::Tensor> outs = {out1, out2};
  Backward(outs, {});

  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 30.0);
}

/*
       inp
        |
      Node0
    ____|____
    |       |
  Node1   Node2
    |       |
   out1    out2
*/
TEST(FwdBwdJoint, GradientHook) {
  eager_test::InitEnv(phi::CPUPlace());

  // 1. Prepare Input
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  egr_utils_api::RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  paddle::Tensor out0 = egr::scale(tensor,
                                   scale0,
                                   bias0,
                                   true /*bias_after_scale*/,
                                   true /*trace_backward*/);
  egr_utils_api::RetainGradForTensor(out0);  // hook: +5
  egr_utils_api::RegisterGradientHookForTensor(out0,
                                               hook_function);  // hook: +5

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  paddle::Tensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);
  egr_utils_api::RetainGradForTensor(out1);  // hook: +5
  egr_utils_api::RegisterGradientHookForTensor(out1,
                                               hook_function);  // hook: +5

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  paddle::Tensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);
  egr_utils_api::RetainGradForTensor(out2);  // hook: +5
  egr_utils_api::RegisterGradientHookForTensor(out2,
                                               hook_function);  // hook: +5

  // 4. Run Backward
  std::vector<paddle::Tensor> outs = {out1, out2};
  Backward(outs, {});

  // Examine Backward Grad
  // leaf grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 190.0);

  // out0 grad
  eager_test::CompareGradTensorWithValue<float>(out0, 90.0);

  // out1 grad
  eager_test::CompareGradTensorWithValue<float>(out1, 1.0);

  // out2 grad
  eager_test::CompareGradTensorWithValue<float>(out2, 1.0);
}

/*
       inp
        |
      Node0
    ____|____
    |       |
  Node1   Node2
    |       |
   out1    out2
*/
TEST(FwdBwdJoint, CrossBatchAccumulation) {
  eager_test::InitEnv(phi::CPUPlace());

  // 1. Prepare Input
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  egr_utils_api::RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  paddle::Tensor out0 = egr::scale(tensor,
                                   scale0,
                                   bias0,
                                   true /*bias_after_scale*/,
                                   true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  paddle::Tensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  paddle::Tensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);

  // 4. Run Backward
  std::vector<paddle::Tensor> outs = {out1, out2};
  Backward(outs, {});

  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 30.0);

  // Cross Batch Accumulation
  Backward(outs, {});

  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 60.0);
}

/* ---------------------------------------------------- */
/* ---------------------- CUDA Tests ------------------ */
/* ---------------------------------------------------- */

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(FwdBwdJoint, SingleNodeCUDA) {
  eager_test::InitEnv(phi::GPUPlace());

  // 1. Prepare Input
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::GPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  egr_utils_api::RetainGradForTensor(tensor);

  // 3. Run Forward
  float scale = 2.0;
  float bias = 3.0;
  paddle::Tensor out = egr::scale(
      tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output
  eager_test::CompareTensorWithValue<float>(out, 13.0);

  std::vector<paddle::Tensor> outs = {out};
  // 4. Run Backward
  Backward(outs, {});

  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 2.0);
}

/*
       inp
        |
      Node0
    ____|____
    |       |
  Node1   Node2
    |       |
   out1    out2
*/
TEST(FwdBwdJoint, BranchedNodesCUDA) {
  eager_test::InitEnv(phi::GPUPlace());

  // 1. Prepare Input
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::GPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);
  egr_utils_api::RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  paddle::Tensor out0 = egr::scale(tensor,
                                   scale0,
                                   bias0,
                                   true /*bias_after_scale*/,
                                   true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  paddle::Tensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  paddle::Tensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output 0
  eager_test::CompareTensorWithValue<float>(out0, 13.0);
  // Examine Forward Output 1
  eager_test::CompareTensorWithValue<float>(out1, 75.0);
  // Examine Forward Output 2
  eager_test::CompareTensorWithValue<float>(out2, 150.0);

  // TODO(jiabin): fix this with add functor
  // 4. Run Backward
  std::vector<paddle::Tensor> outs = {out1, out2};
  Backward(outs, {});

  // Examine Backward Grad
  eager_test::CompareGradTensorWithValue<float>(tensor, 30.0);
}
#endif

}  // namespace egr
