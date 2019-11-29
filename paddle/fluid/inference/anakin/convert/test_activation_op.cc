/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include "paddle/fluid/inference/anakin/convert/activation.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
static void test_activation_op(const std::string& op_type,
                               const platform::DeviceContext& context,
                               bool use_gpu) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);
  validator.DeclInputVar("act-X", {10, 6, 1, 1});
  validator.DeclOutputVar("act-Out", {10, 6, 1, 1});
  framework::OpDesc desc;
  desc.SetType(op_type);
  desc.SetInput("X", {"act-X"});
  desc.SetOutput("Out", {"act-Out"});

  if (op_type == "swish") {
    desc.SetAttr("beta", 1.0f);
  }

  if (op_type == "relu6") {
    desc.SetAttr("threshold", 6.0f);
  }

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(5);
}

#ifdef PADDLE_WITH_CUDA
TEST(sigm_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_activation_op<::anakin::saber::NV>("sigmoid", ctx, true);
}

TEST(tanh_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_activation_op<::anakin::saber::NV>("tanh", ctx, true);
}

TEST(relu6_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_activation_op<::anakin::saber::NV>("relu6", ctx, true);
}

TEST(swish_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_activation_op<::anakin::saber::NV>("swish", ctx, true);
}
#endif

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(sigmoid);
USE_OP(tanh);
USE_OP(relu6);
USE_OP(swish);

USE_ANAKIN_CONVERTER(sigmoid);
USE_ANAKIN_CONVERTER(tanh);
USE_ANAKIN_CONVERTER(relu6);
USE_ANAKIN_CONVERTER(swish);
