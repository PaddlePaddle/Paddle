/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/operators/lite/lite_engine_op.h"
#include "paddle/fluid/operators/lite/ut_helper.h"

USE_NO_KERNEL_OP(lite_engine)

using paddle::inference::lite::AddTensorToBlockDesc;
using paddle::inference::lite::AddFetchListToBlockDesc;
using paddle::inference::lite::CreateTensor;
using paddle::inference::lite::serialize_params;
namespace paddle {
namespace operators {

#if defined(PADDLE_WITH_CUDA)
TEST(LiteEngineOp, engine_op) {
  framework::ProgramDesc program;
  auto* block_ = program.Proto()->mutable_blocks(0);
  framework::BlockDesc block_desc(&program, block_);
  auto* feed0 = block_desc.AppendOp();
  feed0->SetType("feed");
  feed0->SetInput("X", {"feed"});
  feed0->SetOutput("Out", {"x"});
  feed0->SetAttr("col", 0);
  auto* feed1 = block_desc.AppendOp();
  feed1->SetType("feed");
  feed1->SetInput("X", {"feed"});
  feed1->SetOutput("Out", {"y"});
  feed1->SetAttr("col", 1);
  LOG(INFO) << "create elementwise_add op";
  auto* elt_add = block_desc.AppendOp();
  elt_add->SetType("elementwise_add");
  elt_add->SetInput("X", std::vector<std::string>({"x"}));
  elt_add->SetInput("Y", std::vector<std::string>({"y"}));
  elt_add->SetOutput("Out", std::vector<std::string>({"z"}));
  elt_add->SetAttr("axis", -1);
  LOG(INFO) << "create fetch op";
  auto* fetch = block_desc.AppendOp();
  fetch->SetType("fetch");
  fetch->SetInput("X", std::vector<std::string>({"z"}));
  fetch->SetOutput("Out", std::vector<std::string>({"out"}));
  fetch->SetAttr("col", 0);
  // Set inputs' variable shape in BlockDesc
  AddTensorToBlockDesc(block_, "x", std::vector<int64_t>({2, 4}), true);
  AddTensorToBlockDesc(block_, "y", std::vector<int64_t>({2, 4}), true);
  AddTensorToBlockDesc(block_, "z", std::vector<int64_t>({2, 4}), false);
  AddFetchListToBlockDesc(block_, "out");
  *block_->add_ops() = *feed1->Proto();
  *block_->add_ops() = *feed0->Proto();
  *block_->add_ops() = *elt_add->Proto();
  *block_->add_ops() = *fetch->Proto();
  framework::Scope scope;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  platform::CUDAPlace place;
  platform::CUDADeviceContext ctx(place);
  ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                       .GetAllocator(place, ctx.stream())
                       .get());
  ctx.PartialInitWithAllocator();
#else
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
#endif
  // Prepare variables.
  CreateTensor(&scope, "x", std::vector<int64_t>({2, 4}), true);
  CreateTensor(&scope, "y", std::vector<int64_t>({2, 4}), true);
  CreateTensor(&scope, "out", std::vector<int64_t>({2, 4}), false);

  ASSERT_EQ(block_->ops_size(), 4);

  std::vector<std::string> repetitive_params{"x", "y"};
  inference::lite::EngineConfig config;
  config.valid_places = {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    paddle::lite_api::Place({TARGET(kCUDA), PRECISION(kFloat)}),
#endif
    paddle::lite_api::Place({TARGET(kX86), PRECISION(kFloat)}),
    paddle::lite_api::Place({TARGET(kHost), PRECISION(kAny)}),
  };
  serialize_params(&(config.param), &scope, repetitive_params);
  config.model = program.Proto()->SerializeAsString();
  LOG(INFO) << "create lite_engine desc";
  framework::OpDesc engine_op_desc(nullptr);
  engine_op_desc.SetType("lite_engine");
  engine_op_desc.SetInput("Xs", std::vector<std::string>({"x", "y"}));
  engine_op_desc.SetOutput("Ys", std::vector<std::string>({"out"}));
  std::string engine_key = "engine_0";
  engine_op_desc.SetAttr("engine_key", engine_key);
  engine_op_desc.SetAttr("enable_int8", false);
  engine_op_desc.SetAttr("use_gpu", true);
  engine_op_desc.SetAttr("zero_copy", true);
  engine_op_desc.SetBlockAttr("sub_block", &block_desc);
  // TODO(wilber): The ut is out of date, we need to a new lite subgraph test.
  // inference::Singleton<inference::lite::EngineManager>::Global().Create(
  //     engine_key, config);
  // LOG(INFO) << "create engine op";
  // auto engine_op = framework::OpRegistry::CreateOp(engine_op_desc);
  // LOG(INFO) << "engine_op " << engine_op.get();
  // // Execute them.
  // LOG(INFO) << "engine_op run";
  // engine_op->Run(scope, place);
  // LOG(INFO) << "done";
}
#endif

}  // namespace operators
}  // namespace paddle
