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
#include <iostream>

#include "gtest/gtest.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

#include "paddle/fluid/operators/collective/c_allreduce_op.h"
#include "paddle/fluid/operators/collective/gen_hccl_id_op_helper.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;
using float16 = paddle::platform::float16;

USE_OP(c_allreduce_sum);
USE_NO_KERNEL_OP(c_gen_hccl_id);
USE_NO_KERNEL_OP(c_comm_init_hccl);
USE_OP_DEVICE_KERNEL(c_allreduce_sum, NPU);

DECLARE_string(selected_npus);

template <typename T>
void PrintDebugInfo(const std::string preStr, const std::vector<T>& data) {
    std::cout << preStr ; 
  for (auto ele : data) {
      std::cout << ele << ",";
  }
  std::cout << "\n";
}

void PrepareUniqueId(f::Scope* scope, const p::DeviceContext& ctx,
                     HcclRootInfo* hccl_id) {
  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));

  VLOG(2) << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; rank_id = " << rank_id
          << "; RANK_TABLE_FILE = " << atoi(getenv("DEVICE_ID"));

  std::vector<int> rank_ids{0,1};
  f::AttributeMap gen_hccl_id;

  std::vector<std::string> endpointList = {"127.0.0.1:6175", "127.0.0.1:6177"};
  gen_hccl_id["rank"] = rank_id;
  gen_hccl_id["endpoint"] = endpointList[rank_id];
  std::vector<std::string> other_endpoints = {
      endpointList[rank_id == 0 ? 1 : 0]};
  gen_hccl_id["other_endpoints"] = other_endpoints;

  auto out = scope->Var("Out");
  auto id = out->GetMutable<HcclRootInfo>();

  VLOG(3) << "break";

  auto comm_init_op = f::OpRegistry::CreateOp("c_gen_hccl_id", {},
                                              {{"Out", {"Out"}}}, gen_hccl_id);
  VLOG(3) << "break";
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();

  memcpy(hccl_id, id, 1024);
}

void Prepare(f::Scope* scope, const p::DeviceContext& ctx,
             HcclRootInfo* hccl_id) {
  auto x = scope->Var("X");
  auto id = x->GetMutable<HcclRootInfo>();

  memcpy(id, hccl_id, 1024);

  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));

  VLOG(2) << "rank_id = " << rank_id << "; device_id = " << device_id
          << "; rank_id = " << rank_id
          << "; RANK_TABLE_FILE = " << atoi(getenv("DEVICE_ID"));

  f::AttributeMap comm_init_attrs;
  comm_init_attrs["ring_id"] = 0;
  comm_init_attrs["rank_ids"] = 2;
  comm_init_attrs["rank"] = rank_id;
  comm_init_attrs["device_id"] = device_id;
  auto comm_init_op = f::OpRegistry::CreateOp(
      "c_comm_init_hccl", {{"X", {"X"}}}, {}, comm_init_attrs);
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();
}

f::Tensor* alloc_float_status(f::Scope* scope, const p::DeviceContext& ctx){
    auto float_status_var = scope->Var("FloatStatus");
    auto float_status = float_status_var->GetMutable<f::Tensor>();
    float_status->mutable_data<float>(ctx.GetPlace());
    float_status->Resize({8});

    /*
    auto runner = paddle::operators::NpuOpRunner("NPUAllocFloatStatus", {}, {*float_status});
    auto stream = ctx.stream();
    runner.Run(stream);
    */

    VLOG(3) << "before alloc status";
    f::AttributeMap attrs;
    auto op = f::OpRegistry::CreateOp("NPUAllocFloatStatus", {}, {{"FloatStatus", {"FloatStatus"}}}, {});
    VLOG(3) << "after alloc status";

    op->Run(*scope, ctx.GetPlace());
    VLOG(3) << "after alloc status run";

    return float_status;
}

template<typename T>
void TestHCCLAllReduceOp(f::Scope* scope, const p::DeviceContext& ctx,
                         int iter, T val) {
  // init
  auto x = scope->Var("Data");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  int rank_id = atoi(getenv("RANK_ID"));
  int num1 = 3;
  int num2 = 128;

  std::vector<T> init;
  for (int64_t i = 0; i < num1 * num2; ++i) {
    init.push_back(val + rank_id);
  }
  PrintDebugInfo("input data", init);

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({num1, num2});
  ctx.Wait();

  auto out = scope->Var("OutData");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({num1, num2});
  auto place = ctx.GetPlace();
  tensor_out->mutable_data<T>(place);  // allocate
  ctx.Wait();

  // run
  f::AttributeMap attrs;
  attrs["tag"] = std::string("tagx_" + std::to_string(iter));
  attrs["ring_id"] = 0;

  alloc_float_status(scope, ctx);

  auto op = f::OpRegistry::CreateOp("c_allreduce_sum", {{"X", {"Data"}}, {"FloatStatus", {"FloatStatus"}}},
                                    {{"Out", {"OutData"}}}, attrs);

  for (int i = 0; i < 10; i++) {
    op->Run(*scope, place);
  }
  ctx.Wait();

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);
  ctx.Wait();

  PrintDebugInfo("output data", out_vec);

  EXPECT_EQ(out_vec.size(), init.size());
  if(std::isinf(val)){
      for (uint32_t i = 0; i < out_vec.size(); i++) {
        EXPECT_EQ(out_vec[i], 3.0);
      }
  }else{
      for (uint32_t i = 0; i < out_vec.size(); i++) {
        EXPECT_TRUE(!std::isinf(out_vec[i]));
      }
  }
}

TEST(c_allreduce_sum, NPU) {
  f::Scope scope;
  HcclRootInfo hccl_id;

  p::NPUDeviceContext ctx(p::NPUPlace(atoi(FLAGS_selected_npus.c_str())));

  // only support one device, if more than one device, use first default
  PrepareUniqueId(&scope, ctx, &hccl_id);
  Prepare(&scope, ctx, &hccl_id);
  for (int i = 0; i < 1; i++) {
    VLOG(2) << "iter num: " << i;
    TestHCCLAllReduceOp<float>(&scope, ctx, i, 1.0);
  }

  /*
    for (int i = 2; i < 3; i++) {
        VLOG(2) << "iter num: " << i;
        TestHCCLAllReduceOp(&scope, ctx, i);
      }
    for (int i = 3; i < 4; i++) {
        VLOG(2) << "iter num: " << i;
        TestHCCLAllReduceOp(&scope, ctx, i);
      }
      */
}
