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

#include <cmath>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_utils.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"

USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(matmul_v2);
USE_OP_ITSELF(relu);
USE_OP_ITSELF(reduce_mean);
USE_OP_ITSELF(feed);
USE_OP_ITSELF(fetch);
USE_OP_ITSELF(scale);

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(relu, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA)
PD_DECLARE_KERNEL(add, KPS, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(relu, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, GPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace jit {
using DenseTensor = phi::DenseTensor;

std::vector<Tensor> PrepareInputs(const phi::Place& place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(place);

  DenseTensor t;
  t.Resize(phi::make_ddim({2, 4}));
  t.mutable_data<float>(place);
  phi::funcs::set_constant(dev_ctx, &t, 2.);

  return utils::ToTensors({t});
}

TEST(CpuLayerTest, Function) {
  auto func_null = Function();
  EXPECT_TRUE(!func_null.IsValid());
}

TEST(CpuLayerTest, Construct) {
  auto place = phi::CPUPlace();
  std::string path = "./multi_program_load/export";
  auto layer = jit::Load(path, place);

  float fbias = layer.Attribute<float>("fbias");
  EXPECT_FLOAT_EQ(fbias, 1.4);

  int ds = layer.Attribute<int>("down_sampling");
  EXPECT_EQ(ds, 4);

  std::string fstr = layer.Attribute<std::string>("fstr");
  EXPECT_STREQ(fstr.c_str(), "save str property");

  std::vector<int> ints = layer.Attribute<std::vector<int>>("ints");
  EXPECT_EQ(ints[0], 10);
  EXPECT_EQ(ints[1], 20);

  std::vector<float> floats = layer.Attribute<std::vector<float>>("floats");
  EXPECT_FLOAT_EQ(floats[0], 1.1);
  EXPECT_FLOAT_EQ(floats[1], 2.2);

  std::vector<std::string> strs =
      layer.Attribute<std::vector<std::string>>("strs");
  EXPECT_STREQ(strs[0].c_str(), "hello");
  EXPECT_STREQ(strs[1].c_str(), "world");

  // functions
  auto inputs = PrepareInputs(place);
  auto outs = layer.forward(inputs);
  auto out_data = outs[0].data<float>();
  EXPECT_NEAR(out_data[0], 0.02194316, 1e-6);

  auto func = layer.Function("infer");
  EXPECT_TRUE(func.IsValid());
  outs = func(inputs);
  out_data = outs[0].data<float>();
  EXPECT_NEAR(out_data[0], 1.41562390, 1e-6);
  auto pow_out =
      paddle::experimental::pow(outs[0], paddle::experimental::Scalar(2));
  out_data = pow_out.data<float>();
  EXPECT_NEAR(out_data[0], pow(1.41562390, 2.0), 1e-6);
}

#if defined(PADDLE_WITH_CUDA)
TEST(GpuLayerTest, Construct) {
  auto place = phi::GPUPlace();

  std::string path = "./multi_program_load/export";
  auto layer = jit::Load(path, place);
  auto inputs = PrepareInputs(place);

  auto outs = layer.forward(inputs);
  auto gpu_tensor = outs[0];
  auto cpu_tensor =
      paddle::experimental::copy_to(gpu_tensor, phi::CPUPlace(), true);
  auto out_data = cpu_tensor.data<float>();
  EXPECT_NEAR(out_data[0], 0.02194316, 1e-6);

  auto func = layer.Function("infer");
  EXPECT_TRUE(func.IsValid());
  outs = func(inputs);
  gpu_tensor = outs[0];
  cpu_tensor = paddle::experimental::copy_to(gpu_tensor, phi::CPUPlace(), true);
  out_data = cpu_tensor.data<float>();
  EXPECT_NEAR(out_data[0], 1.41562390, 1e-6);

  auto sqrt_out = paddle::experimental::sqrt(outs[0]);
  cpu_tensor = paddle::experimental::copy_to(sqrt_out, phi::CPUPlace(), true);
  out_data = cpu_tensor.data<float>();
  EXPECT_NEAR(out_data[0], sqrt(1.41562390), 1e-6);
}
#endif

}  // namespace jit
}  // namespace paddle
