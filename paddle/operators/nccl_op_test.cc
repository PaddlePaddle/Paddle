/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#include "paddle/operators/nccl_op.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/gpu_info.h"

#include <thrust/device_vector.h>
#include <memory>
#include <vector>

static std::vector<int> gpu_list;

using f = paddle::framework;
using ops = paddle::operators;

void AddOp(const std::string &type, const f::VariableNameMap &inputs,
           const f::VariableNameMap &outputs, f::AttributeMap attrs,
           paddle::framework::BlockDescBind *block) {
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      auto var = block->Var(v);
      var->SetDataType(paddle::framework::DataType::FP32);
    }
  }

  auto op = block->AppendOp();
  op->SetType(type);
  for (auto &kv : inputs) {
    op->SetInput(kv.first, kv.second);
  }
  for (auto &kv : outputs) {
    op->SetOutput(kv.first, kv.second);
  }
  op->SetAttrMap(attrs);
}

TEST(NCCL, ncclInitOp) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.Block(0);
}

int main(int argc, char **argv) {
  static constexpr int gpu_count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < gpu_count; ++i) {
    gpu_list.emplace_back(i);
  }
  if (dev_count <= 1) {
    LOG(WARNING)
        << "Cannot test multi-gpu nccl, because the CUDA device count is "
        << dev_count;
    return 0;
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
