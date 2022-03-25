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

#include <gtest/gtest.h>

#include "paddle/fluid/inference/utils/singleton.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

#include "paddle/fluid/inference/lite/engine.h"
#include "paddle/fluid/operators/lite/ut_helper.h"

namespace paddle {
namespace inference {
namespace lite {

using inference::lite::AddTensorToBlockDesc;
using paddle::inference::lite::AddFetchListToBlockDesc;
using inference::lite::CreateTensor;
using inference::lite::serialize_params;

void make_fake_model(std::string* model, std::string* param) {
  framework::ProgramDesc program;
  LOG(INFO) << "program.block size is " << program.Size();
  auto* block_ = program.Proto()->mutable_blocks(0);
  LOG(INFO) << "create block desc";
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

  *block_->add_ops() = *feed0->Proto();
  *block_->add_ops() = *feed1->Proto();
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
  std::vector<std::string> repetitive_params{"x", "y"};
  CreateTensor(&scope, "x", std::vector<int64_t>({2, 4}));
  CreateTensor(&scope, "y", std::vector<int64_t>({2, 4}));
  ASSERT_EQ(block_->ops_size(), 4);
  *model = program.Proto()->SerializeAsString();
  serialize_params(param, &scope, repetitive_params);
}

TEST(EngineManager, engine) {
  ASSERT_EQ(
      inference::Singleton<inference::lite::EngineManager>::Global().Empty(),
      true);

  inference::lite::EngineConfig config;
  make_fake_model(&(config.model), &(config.param));
  LOG(INFO) << "prepare config";

  const std::string unique_key("engine_0");
  config.model_from_memory = true;
  config.valid_places = {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    paddle::lite_api::Place({TARGET(kCUDA), PRECISION(kFloat)}),
#endif
    paddle::lite_api::Place({TARGET(kX86), PRECISION(kFloat)}),
    paddle::lite_api::Place({TARGET(kHost), PRECISION(kAny)}),
  };

  LOG(INFO) << "Create EngineManager";
  // TODO(wilber): The ut is out of date, we need to a new lite subgraph test.
  // inference::Singleton<inference::lite::EngineManager>::Global().Create(
  //     unique_key, config);
  // LOG(INFO) << "Create EngineManager done";
  // ASSERT_EQ(
  //     inference::Singleton<inference::lite::EngineManager>::Global().Empty(),
  //     false);
  // ASSERT_EQ(inference::Singleton<inference::lite::EngineManager>::Global().Has(
  //               unique_key),
  //           true);
  // paddle::lite_api::PaddlePredictor* engine_0 =
  //     inference::Singleton<inference::lite::EngineManager>::Global().Get(
  //         unique_key);
  // CHECK_NOTNULL(engine_0);
  // inference::Singleton<inference::lite::EngineManager>::Global().DeleteAll();
  // CHECK(inference::Singleton<inference::lite::EngineManager>::Global().Get(
  //           unique_key) == nullptr)
  //     << "the engine_0 should be nullptr";
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle
