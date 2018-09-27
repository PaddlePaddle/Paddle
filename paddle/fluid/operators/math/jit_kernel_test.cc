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

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(JitKernel, pool) {
  namespace jit = paddle::operators::math::jitkernel;
  const int frame_size = 4;
  std::string act_gate = "sigmoid", act_cand = "tanh", act_cell = "tanh";
  const auto& plstm1 =
      jit::KernelPool::Instance()
          .template Get<jit::LSTMKernel<float>, int, const std::string&,
                        const std::string&, const std::string&>(
              frame_size, act_gate, act_cand, act_cell);
  const auto& plstm2 =
      jit::KernelPool::Instance()
          .template Get<jit::LSTMKernel<float>, int, const std::string&,
                        const std::string&, const std::string&>(
              frame_size, act_gate, act_cand, act_cell);
  EXPECT_EQ(plstm1, plstm2);

  const auto& pvmul_f =
      jit::KernelPool::Instance().template Get<jit::VMulKernel<float>>(4);
  EXPECT_TRUE(std::dynamic_pointer_cast<jit::Kernel>(plstm2) !=
              std::dynamic_pointer_cast<jit::Kernel>(pvmul_f));

  const auto& pvmul_d =
      jit::KernelPool::Instance().template Get<jit::VMulKernel<double>>(4);
  EXPECT_TRUE(std::dynamic_pointer_cast<jit::Kernel>(pvmul_f) !=
              std::dynamic_pointer_cast<jit::Kernel>(pvmul_d));

  const auto& pvmul_from_key = jit::KernelPool::Instance().Get("vmulf4");
  EXPECT_TRUE(pvmul_f == pvmul_from_key);
  const auto& pvmul_from_key2 = jit::KernelPool::Instance().Get("vmulf5");
  EXPECT_TRUE(pvmul_from_key2 == nullptr);
}
