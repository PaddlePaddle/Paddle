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
#include <fstream>
#include <ios>

#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"

#include "paddle/fluid/inference/lite/engine.h"
#include "paddle/fluid/inference/utils/singleton.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace lite {

namespace {

void AddTensorToBlockDesc(framework::proto::BlockDesc* block,
                          const std::string& name,
                          const std::vector<int64_t>& shape) {
  using framework::proto::VarType;
  auto* var = block->add_vars();
  framework::VarDesc desc(name);
  desc.SetType(VarType::LOD_TENSOR);
  desc.SetDataType(VarType::FP32);
  desc.SetShape(shape);
  *var = *desc.Proto();
}

void make_fake_model(std::string* model, std::string* param) {
  framework::ProgramDesc program;
  auto* block_ = program.Proto()->mutable_blocks(0);
  LOG(INFO) << "create block desc";
  framework::BlockDesc block_desc(&program, block_);
  LOG(INFO) << "create feed op";
  auto* feed0 = block_desc.AppendOp();
  feed0->SetType("feed");
  feed0->SetInput("X", {"feed"});
  feed0->SetOutput("Out", {"x"});
  feed0->SetAttr("col", 1);
  AddTensorToBlockDesc(block_, "x", std::vector<int64_t>({2, 4, 1, 1}));
  *block_->add_ops() = *feed0->Proto();
  ASSERT_EQ(block_->ops_size(), 1);
  framework::Scope scope;
  platform::CPUPlace place;
  platform::CPUDeviceContext ctx(place);
  *model = program.Proto()->SerializeAsString();
}

}  // namespace

TEST(EngineManager, manual) {
  ASSERT_EQ(
      inference::Singleton<inference::lite::EngineManager>::Global().Empty(),
      true);

  inference::lite::EngineConfig config;
  make_fake_model(&(config.model), &(config.param));

  const std::string unique_key("engine_0");
  config.model_from_memory = true;
  config.prefer_place = {TARGET(kX86), PRECISION(kFloat)};
  config.valid_places = {
      paddle::lite::Place({TARGET(kX86), PRECISION(kFloat)}),
      paddle::lite::Place({TARGET(kHost), PRECISION(kAny)}),
#ifdef PADDLE_WITH_CUDA
      paddle::lite::Place({TARGET(kCUDA), PRECISION(kFloat)}),
#endif
  };

  LOG(INFO) << "Create EngineManager";
  inference::Singleton<inference::lite::EngineManager>::Global().Create(
      unique_key, config);
  LOG(INFO) << "Create EngineManager done";
  ASSERT_EQ(
      inference::Singleton<inference::lite::EngineManager>::Global().Empty(),
      false);
  ASSERT_EQ(inference::Singleton<inference::lite::EngineManager>::Global().Has(
                unique_key),
            true);
  paddle::lite::Predictor* engine_0 =
      inference::Singleton<inference::lite::EngineManager>::Global().Get(
          unique_key);

  CHECK_NOTNULL(engine_0);
  inference::Singleton<inference::lite::EngineManager>::Global().DeleteAll();
  CHECK(inference::Singleton<inference::lite::EngineManager>::Global().Get(
            unique_key) == nullptr)
      << "the engine_0 should be nullptr";
}

}  // namespace lite
}  // namespace paddle
