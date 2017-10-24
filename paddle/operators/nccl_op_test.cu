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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <memory>
#include <vector>

#include "paddle/framework/block_desc.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/program_desc.h"
#include "paddle/framework/var_desc.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/gpu_info.h"
#include "paddle/platform/place.h"

USE_CPU_ONLY_OP(ncclInit);
USE_GPU_ONLY_OP(ncclAllReduce);
USE_GPU_ONLY_OP(ncclReduce);
USE_GPU_ONLY_OP(ncclBcastSend);
USE_GPU_ONLY_OP(ncclBcastRecv);

static std::vector<int> gpu_list;

namespace f = paddle::framework;
namespace ops = paddle::operators;

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

// ncclInitOp with desc
TEST(NCCL, ncclInitOp) {
  f::ProgramDescBind program;
  f::BlockDescBind *block = program.Block(0);
  f::OpDescBind *op1 = block->AppendOp();

  op1->SetType("ncclInit");
  op1->SetOutput("Communicator", {"x1"});
  op1->SetAttr("gpus", {gpu_list});
  f::Scope g_scope;
  paddle::platform::DeviceContext *ctx =
      new paddle::platform::CPUDeviceContext(paddle::platform::CPUPlace());

  auto *var = g_scope.Var("x1");
  var->GetMutable<paddle::platform::Communicator>();

  auto op = f::OpRegistry::CreateOp(*op1);
  VLOG(1) << "invoke NCCLInitOp.";
  op->Run(g_scope, *ctx);
  VLOG(1) << "NCCLInitOp finished.";
}

int main(int argc, char **argv) {
  static int dev_count = paddle::platform::GetCUDADeviceCount();
  if (dev_count <= 1) {
    LOG(WARNING)
        << "Cannot test multi-gpu nccl, because the CUDA device count is "
        << dev_count;
    return 0;
  }

  for (int i = 0; i < dev_count; ++i) {
    gpu_list.emplace_back(i);
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
