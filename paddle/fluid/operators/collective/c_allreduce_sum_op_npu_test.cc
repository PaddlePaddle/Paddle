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

#include <string>
#include <thread>  // NOLINT
#include <vector>
#include <stdio.h>

#include "gtest/gtest.h"

#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/operators/collective/c_allreduce_op.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(c_allreduce_sum);
USE_NO_KERNEL_OP(c_comm_init_hcom);
USE_OP_DEVICE_KERNEL(c_allreduce_sum, NPU);

DECLARE_string(selected_npus);

template<typename T>
void PrintDebugInfo(const std::string preStr, const std::vector<T> &data){
  std::string debugstring = "";
  for (auto ele : data) {
    debugstring += std::to_string(ele) + std::string(",");
  }
  VLOG(2) << preStr << ":" << std::endl <<debugstring; 
}

void Prepare(f::Scope* scope, const p::DeviceContext& ctx){

  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));

  VLOG(2) << "rank_id = " << rank_id
  << "; device_id = " << device_id  
  << "; rank_id = " << rank_id  
  << "; RANK_TABLE_FILE = " << atoi(getenv("RANK_TABLE_FILE"));  
  
  std::vector<int> rank_ids{0, 1};
  f::AttributeMap comm_init_attrs;
  comm_init_attrs["ring_id"] = 0;
  comm_init_attrs["nranks"] = 2;
  comm_init_attrs["rank"] = rank_id;
  comm_init_attrs["device_id"] = device_id;
  comm_init_attrs["rank_ids"] = rank_ids;
  auto comm_init_op =
      f::OpRegistry::CreateOp("c_comm_init_hcom", {}, {}, comm_init_attrs);
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();
}

void TestHCCLAllReduceOp(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  int rank_id = atoi(getenv("RANK_ID"));
  int num1 = 3;
  int num2 = 128;

  std::vector<float> init;
  for (int64_t i = 0; i < num1 * num2; ++i) {
    init.push_back(1.0 + rank_id);
  }
  PrintDebugInfo("input data", init);

  auto place = ctx.GetPlace();

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({num1, num2});
  ctx.Wait();

  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({num1, num2});
  tensor_out->mutable_data<float>(place);  // allocate
  ctx.Wait();

  // run
  f::AttributeMap attrs;
  attrs["tag"]=std::string("tagx");
  attrs["ring_id"]=0;

  auto op = f::OpRegistry::CreateOp("c_allreduce_sum", 
                                    {{"X", {"X"}}},
                                    {{"Out", {"Out"}}}, 
                                    attrs);

  op->Run(*scope, place);
  ctx.Wait();

  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);
  ctx.Wait();

  PrintDebugInfo("output data", out_vec);

  EXPECT_EQ(out_vec.size(), init.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], 3.0);
  }
}

TEST(c_allreduce_sum, NPU) {
  f::Scope scope;

  // only support one device, if more than one device, use first default  
  p::NPUDeviceContext ctx(p::NPUPlace(atoi(FLAGS_selected_npus.c_str())));

  Prepare(&scope, ctx);
  TestHCCLAllReduceOp(&scope, ctx);
}
