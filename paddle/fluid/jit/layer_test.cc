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

#include "paddle/fluid/jit/layer.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

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

namespace paddle {
namespace jit {

std::vector<Variable> PrepareInputs() {
  auto temp = DenseTensor();
  temp.Resize(phi::make_ddim({2, 4}));
  phi::CPUContext cpu_ctx;
  cpu_ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                           .GetAllocator(paddle::platform::CPUPlace())
                           .get());
  cpu_ctx.Init();
  cpu_ctx.Alloc<float>(&temp);
  phi::funcs::set_constant(cpu_ctx, &temp, 2.);
  Variable v;
  auto *p = v.GetMutable<DenseTensor>();
  *p = temp;
  // TODO(dev): associate the input name
  return {v};
}

TEST(layer, Construct) {
  std::string path = "./Testing/";
  auto layer = jit::Load(path);
  auto inputs = PrepareInputs();

  auto outs = layer.forward(inputs);
  auto out_vars = outs[0];
  auto out_dense_tensor = out_vars.Get<DenseTensor>();
  auto out_data = out_dense_tensor.data<float>();
  EXPECT_NEAR(out_data[0], 0.02194316, 1e-6);

  auto func = layer.GetFunction("infer");
  outs = (*func)(inputs);
  out_vars = outs[0];
  out_dense_tensor = out_vars.Get<DenseTensor>();
  out_data = out_dense_tensor.data<float>();
  EXPECT_NEAR(out_data[0], 1.41562390, 1e-6);
}

}  // namespace jit
}  // namespace paddle
