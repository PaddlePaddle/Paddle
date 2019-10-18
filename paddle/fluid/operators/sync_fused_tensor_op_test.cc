// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/sync_fused_tensor_op.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_memory_aligment.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
using LoDTensor = paddle::framework::LoDTensor;
using Place = paddle::platform::Place;
using Scope = paddle::framework::Scope;
using CPUPlace = paddle::platform::CPUPlace;
using CUDAPlace = paddle::platform::CUDAPlace;

USE_CUDA_ONLY_OP(sync_fused_tensor);

void InitTensorData(LoDTensor *tensor, const Place &place) {
  int *data = tensor->mutable_data<int>(place);
  if (platform::is_cpu_place(place)) {
    for (int64_t i = 0; i < tensor->numel(); ++i) {
      data[i] = static_cast<int>(i);
    }
  }
}

size_t GetMemorySize(const std::vector<const LoDTensor *> &in_tensors,
                     const size_t &dtype_size, const Place &place) {
  size_t numel = 0;
  for (size_t i = 0; i < in_tensors.size(); ++i) {
    size_t size = static_cast<size_t>(in_tensors[i]->numel());
    numel += platform::Alignment(size * dtype_size, place) / dtype_size;
  }
  return numel;
}

void CreateAndRunOp(const Scope &scope, const Place &place) {
  auto sync_fused_tensor_op = framework::OpRegistry::CreateOp(
      "sync_fused_tensor", {{"Input", {"input_var_1", "input_var_2"}},
                            {"FusedInput", {"fused_var"}}},
      {{"FusedOutput", {"fused_var"}}}, {});
  sync_fused_tensor_op->Run(scope, place);
}

void FuseTensors(const std::vector<const LoDTensor *> &in_tensors,
                 LoDTensor *fused_tensor, const Place &place,
                 bool same_place_ctrl = false) {
  size_t dtype_size = framework::SizeOfType(framework::proto::VarType::INT32);
  fused_tensor
      ->Resize(framework::make_ddim(
          {static_cast<int64_t>(GetMemorySize(in_tensors, dtype_size, place))}))
      .mutable_data<int>(place);
  size_t offset = 0;
  for (size_t i = 0; i < in_tensors.size(); ++i) {
    size_t len = static_cast<size_t>(in_tensors[i]->numel());
    auto sub_tensor = fused_tensor->Slice(static_cast<int64_t>(offset),
                                          static_cast<int64_t>(offset + len));
    if (same_place_ctrl) {
      if (platform::is_same_place(in_tensors[i]->place(), place)) {
        framework::TensorCopy(*in_tensors[i], place, &sub_tensor);
      }
    } else {
      framework::TensorCopy(*in_tensors[i], place, &sub_tensor);
    }
    offset += platform::Alignment(len * dtype_size, place) / dtype_size;
  }
}

LoDTensor *CopyTensorToCPU(Scope *scope, const LoDTensor &src_tensor,
                           const std::string &dst_var_name) {
  auto dst_tensor = scope->Var(dst_var_name)->GetMutable<LoDTensor>();
  dst_tensor->Resize(framework::make_ddim({src_tensor.numel()}))
      .mutable_data<int>(CPUPlace());
  framework::TensorCopy(src_tensor, CPUPlace(), dst_tensor);
  return dst_tensor;
}

bool CompareOutputTensors(LoDTensor *expect, LoDTensor *actual) {
  int *expect_data = expect->data<int>();
  int *actual_data = actual->data<int>();
  for (int64_t i = 0; i < expect->numel(); ++i) {
    if (expect_data[i] != actual_data[i]) {
      return false;
    }
  }
  return true;
}

// correct
TEST(SyncFusedTensorOp, SyncCPUToGPU) {
  Scope scope;
  CPUPlace cpu_place;
  CUDAPlace gpu_place;

  // prepare data
  auto in_var_1 = scope.Var("input_var_1");
  auto in_var_2 = scope.Var("input_var_2");
  auto fused_var = scope.Var("fused_var");

  // init input tensors
  auto in_tensor_1 = in_var_1->GetMutable<LoDTensor>();
  auto in_tensor_2 = in_var_2->GetMutable<LoDTensor>();
  in_tensor_1->Resize({3, 10});
  in_tensor_2->Resize({20, 5});
  InitTensorData(in_tensor_1, cpu_place);
  InitTensorData(in_tensor_2, gpu_place);

  // init fused input and output tensor
  std::vector<const LoDTensor *> in_tensors;
  in_tensors.emplace_back(in_tensor_1);
  in_tensors.emplace_back(in_tensor_2);
  auto fused_tensor = fused_var->GetMutable<LoDTensor>();
  FuseTensors(in_tensors, fused_tensor, gpu_place, true);

  // prepare expect output tensor
  auto expect_tensor = scope.Var("expect")->GetMutable<LoDTensor>();
  FuseTensors(in_tensors, expect_tensor, gpu_place, false);

  // compare output
  CreateAndRunOp(scope, gpu_place);
  auto fused_tensor_cpu = CopyTensorToCPU(&scope, *fused_tensor, "fused_cpu");
  auto expect_tensor_cpu =
      CopyTensorToCPU(&scope, *expect_tensor, "expect_cpu");
  EXPECT_TRUE(CompareOutputTensors(fused_tensor_cpu, expect_tensor_cpu));
}
