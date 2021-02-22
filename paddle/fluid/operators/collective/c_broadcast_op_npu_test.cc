/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef _WIN32
#include <unistd.h>
#endif

#include <stdio.h>

#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/operators/collective/c_broadcast_op.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(c_broadcast);
USE_NO_KERNEL_OP(c_comm_init_hccl);
USE_OP_DEVICE_KERNEL(c_broadcast, NPU);

void Prepare(f::Scope* scope, const p::DeviceContext& ctx){

  std::string rank_table_file = getenv("RANK_TABLE_FILE");
  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));
  
  printf("rank_table_file: %s, rank_id = %d, device_id = %d\n", rank_table_file.c_str(), rank_id, device_id);
  
  f::AttributeMap attrs;
  attrs["rank_table_file"]=rank_table_file;
  attrs["rank_id"]=rank_id;
  attrs["device_id"]=device_id;
  
  auto op =
      f::OpRegistry::CreateOp("c_comm_init_hccl", {}, {}, attrs);

  auto place = ctx.GetPlace();
  op->Run(*scope, place);

  ctx.Wait();
}
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  std::vector<float> init;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init.push_back(1.0);
  }

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({10, 10});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({10, 10});
  tensor_out->mutable_data<float>(place);  // allocate

  // run
  f::AttributeMap attrs;
  auto op =
      f::OpRegistry::CreateOp("c_broadcast", {{"X", {"X"}}},
                              {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);

  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  EXPECT_EQ(out_vec.size(), init.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], 1.0);
  }
}

TEST(c_broadcast, NPU) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Prepare(&scope, ctx);
  Compare(&scope, ctx);
}
