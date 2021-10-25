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

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/nodes/accumulation_node.h"
#include "paddle/fluid/eager/nodes/scale_node.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_meta.h"

#include "paddle/fluid/eager/tests/test_utils.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

egr::EagerTensor hook_function(const egr::EagerTensor& t) {
  auto t_dense = std::dynamic_pointer_cast<pten::DenseTensor>(t.impl());

  auto ret_meta = pten::TensorMeta(t_dense->dims(), t_dense->backend(),
                                   t_dense->data_type(), t_dense->layout());
  auto ret_dense = std::make_shared<pten::DenseTensor>(std::move(ret_meta),
                                                       pten::TensorStatus());

  float* t_ptr = t_dense->mutable_data<float>();
  float* ret_ptr = ret_dense->mutable_data<float>();
  for (int i = 0; i < ret_dense->numel(); i++) {
    ret_ptr[i] = t_ptr[i] + 5.0;
  }

  auto ret_impl = std::dynamic_pointer_cast<pten::TensorBase>(ret_dense);
  egr::EagerTensor ret = egr::EagerTensor();
  ret.set_impl(ret_impl);

  return ret;
}

TEST(FwdBwdJoint, SingleNode) {
  InitEnv(paddle::platform::CPUPlace());

  // 1. Prepare Input
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CPU, pten::DataType::FLOAT32, pten::DataLayout::NCHW,
      5.0 /*value*/, true /*is_leaf*/);
  RetainGradForTensor(tensor);

  // 3. Run Forward
  float scale = 2.0;
  float bias = 3.0;
  egr::EagerTensor out = egr::scale(
      tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output
  CompareTensorWithValue<float>(out, 13.0);

  std::vector<egr::EagerTensor> outs = {out};
  // 4. Run Backward
  RunBackward(outs, {});

  VLOG(7) << "Target Grad is: "
          << std::static_pointer_cast<pten::DenseTensor>(
                 EagerUtils::unsafe_autograd_meta(tensor)->Grad().impl())
                 ->data<float>()[0];
  // Examine Backward Grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 2.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 2.0));
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
  InitEnv(paddle::platform::CPUPlace());

  // 1. Prepare Input
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CPU, pten::DataType::FLOAT32, pten::DataLayout::NCHW,
      5.0 /*value*/, true /*is_leaf*/);
  RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  egr::EagerTensor out0 =
      egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/,
                 true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  egr::EagerTensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output 0
  CompareTensorWithValue<float>(out0, 13.0);

  // Examine Forward Output 1
  CompareTensorWithValue<float>(out1, 75.0);

  std::vector<egr::EagerTensor> outs = {out1};
  // 4. Run Backward
  RunBackward(outs, {});

  // Examine Backward Grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 10.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 10.0));
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
  InitEnv(paddle::platform::CPUPlace());

  // 1. Prepare Input
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CPU, pten::DataType::FLOAT32, pten::DataLayout::NCHW,
      5.0 /*value*/, true /*is_leaf*/);
  RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  egr::EagerTensor out0 =
      egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/,
                 true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  egr::EagerTensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  egr::EagerTensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output 0
  CompareTensorWithValue<float>(out0, 13.0);

  // Examine Forward Output 1
  CompareTensorWithValue<float>(out1, 75.0);

  // Examine Forward Output 2
  {
    auto dense_out = std::dynamic_pointer_cast<pten::DenseTensor>(out2.impl());
    float* ptr = dense_out->mutable_data<float>();
    for (int i = 0; i < 20; i++) {
      PADDLE_ENFORCE(
          ptr[i] == 150.0,
          paddle::platform::errors::Fatal(
              "Numerical Error, Expected %f but got %f", 150.0, ptr[i]));
    }
  }

  // 4. Run Backward
  std::vector<egr::EagerTensor> outs = {out1, out2};
  RunBackward(outs, {});

  // Examine Backward Grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 30.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 30.0));
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
  InitEnv(paddle::platform::CPUPlace());

  // 1. Prepare Input
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CPU, pten::DataType::FLOAT32, pten::DataLayout::NCHW,
      5.0 /*value*/, true /*is_leaf*/);
  RetainGradForTensor(tensor);

  std::function<egr::EagerTensor(const egr::EagerTensor&)> hook =
      &hook_function;

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  egr::EagerTensor out0 =
      egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/,
                 true /*trace_backward*/);
  RetainGradForTensor(out0);                  // hook: +5
  RegisterGradientHookForTensor(out0, hook);  // hook: +5

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  egr::EagerTensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);
  RetainGradForTensor(out1);                  // hook: +5
  RegisterGradientHookForTensor(out1, hook);  // hook: +5

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  egr::EagerTensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);
  RetainGradForTensor(out2);                  // hook: +5
  RegisterGradientHookForTensor(out2, hook);  // hook: +5

  // 4. Run Backward
  std::vector<egr::EagerTensor> outs = {out1, out2};
  RunBackward(outs, {});

  // Examine Backward Grad
  // leaf grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 190.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 190.0));

  // out0 grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(out0, 90.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 90.0));

  // out1 grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(out1, 1.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));

  // out2 grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(out2, 1.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));
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
  InitEnv(paddle::platform::CPUPlace());

  // 1. Prepare Input
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CPU, pten::DataType::FLOAT32, pten::DataLayout::NCHW,
      5.0 /*value*/, true /*is_leaf*/);
  RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  egr::EagerTensor out0 =
      egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/,
                 true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  egr::EagerTensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  egr::EagerTensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);

  // 4. Run Backward
  std::vector<egr::EagerTensor> outs = {out1, out2};
  RunBackward(outs, {});

  // Examine Backward Grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 30.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 30.0));

  // Cross Batch Accumulation
  RunBackward(outs, {});

  // Examine Backward Grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 60.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 60.0));
}

/* ---------------------------------------------------- */
/* ---------------------- CUDA Tests ------------------ */
/* ---------------------------------------------------- */

TEST(FwdBwdJoint, SingleNodeCUDA) {
  InitEnv(paddle::platform::CUDAPlace());

  // 1. Prepare Input
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CUDA, pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 5.0 /*value*/, true /*is_leaf*/);
  RetainGradForTensor(tensor);

  // 3. Run Forward
  float scale = 2.0;
  float bias = 3.0;
  egr::EagerTensor out = egr::scale(
      tensor, scale, bias, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output
  CompareTensorWithValue<float>(out, 13.0);

  std::vector<egr::EagerTensor> outs = {out};
  // 4. Run Backward
  RunBackward(outs, {});

  // Examine Backward Grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 2.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 2.0));
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
  InitEnv(paddle::platform::CUDAPlace());

  // 1. Prepare Input
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CUDA, pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 5.0 /*value*/, true /*is_leaf*/);
  RetainGradForTensor(tensor);

  // 3. Run Forward
  // Run Forward Node 0
  float scale0 = 2.0;
  float bias0 = 3.0;
  egr::EagerTensor out0 =
      egr::scale(tensor, scale0, bias0, true /*bias_after_scale*/,
                 true /*trace_backward*/);

  // Run Forward Node 1
  float scale1 = 5.0;
  float bias1 = 10.0;
  egr::EagerTensor out1 = egr::scale(
      out0, scale1, bias1, true /*bias_after_scale*/, true /*trace_backward*/);

  // Run Forward Node 2
  float scale2 = 10.0;
  float bias2 = 20.0;
  egr::EagerTensor out2 = egr::scale(
      out0, scale2, bias2, true /*bias_after_scale*/, true /*trace_backward*/);

  // Examine Forward Output 0
  CompareTensorWithValue<float>(out0, 13.0);
  // Examine Forward Output 1
  CompareTensorWithValue<float>(out1, 75.0);
  // Examine Forward Output 2
  CompareTensorWithValue<float>(out2, 150.0);

  // TODO(jiabin): fix this with add functor
  // 4. Run Backward
  std::vector<egr::EagerTensor> outs = {out1, out2};
  RunBackward(outs, {});

  // Examine Backward Grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 30.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 30.0));
}
