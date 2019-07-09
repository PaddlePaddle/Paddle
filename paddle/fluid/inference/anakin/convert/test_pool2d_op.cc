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
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void test_pool2d(const platform::DeviceContext& context, bool use_gpu,
                 bool global_pooling, bool ceil_mode,
                 std::string pool_type = "max") {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);

  // The ITensor's Dims should not contain the batch size.
  // So, the ITensor's Dims of input and output should be C * H * W.
  validator.DeclInputVar("pool2d_x", {1, 3, 6, 7});
  if (global_pooling)
    validator.DeclOutputVar("pool2d_out", {1, 3, 1, 1});
  else if (ceil_mode)
    validator.DeclOutputVar("pool2d_out", {1, 3, 3, 4});
  else
    validator.DeclOutputVar("pool2d_out", {1, 3, 3, 3});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("pool2d");
  desc.SetInput("X", {"pool2d_x"});
  desc.SetOutput("Out", {"pool2d_out"});

  std::vector<int> ksize({2, 2});
  std::vector<int> strides({2, 2});
  std::vector<int> paddings({0, 0});
  std::string pooling_t = pool_type;

  desc.SetAttr("pooling_type", pooling_t);
  desc.SetAttr("ksize", ksize);
  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("global_pooling", global_pooling);
  desc.SetAttr("ceil_mode", ceil_mode);

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(1);
}

#ifdef PADDLE_WITH_CUDA
TEST(Pool2dOpConverter, normal) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_pool2d<::anakin::saber::NV>(ctx, true, false, false);
}
TEST(Pool2dOpConverter, test_global_pooling) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_pool2d<::anakin::saber::NV>(ctx, true, true, false);
}

TEST(Pool2dOpConverter, max_ceil_test) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_pool2d<::anakin::saber::NV>(ctx, true, false, true);
}

TEST(Pool2dOpConverter, avg_ceil_test) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_pool2d<::anakin::saber::NV>(ctx, true, false, true, "avg");
}
#endif
#ifdef ANAKIN_X86_PLACE
TEST(Pool2dOpConverter, normal_cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_pool2d<::anakin::saber::X86>(ctx, false, false, false);
}
TEST(Pool2dOpConverter, test_global_pooling_cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_pool2d<::anakin::saber::X86>(ctx, false, true, false);
}

TEST(Pool2dOpConverter, max_ceil_test_cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_pool2d<::anakin::saber::X86>(ctx, false, false, true);
}

TEST(Pool2dOpConverter, avg_ceil_test_cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_pool2d<::anakin::saber::X86>(ctx, false, false, true, "avg");
}
#endif
}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(pool2d);
USE_ANAKIN_CONVERTER(pool2d);
