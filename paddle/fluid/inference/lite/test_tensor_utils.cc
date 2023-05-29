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

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/lite/engine.h"
#include "paddle/fluid/inference/lite/tensor_utils.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/operators/lite/ut_helper.h"

namespace paddle {
namespace inference {
namespace lite {
namespace utils {

using inference::lite::AddTensorToBlockDesc;
using inference::lite::CreateTensor;
using inference::lite::serialize_params;
using paddle::inference::lite::AddFetchListToBlockDesc;
using paddle::lite_api::DataLayoutType;
using paddle::lite_api::PrecisionType;
using paddle::lite_api::TargetType;

TEST(LiteEngineOp, GetNativePlace) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  platform::Place GetNativePlace(const TargetType& type, int id = 0);
  EXPECT_TRUE(platform::is_cpu_place(GetNativePlace(TargetType::kHost)));
  EXPECT_TRUE(platform::is_gpu_place(GetNativePlace(TargetType::kCUDA)));
  EXPECT_ANY_THROW(GetNativePlace(TargetType::kUnk));
}

TEST(LiteEngineOp, GetLiteTargetType) {
  TargetType GetLiteTargetType(const platform::Place& place);
  ASSERT_EQ(GetLiteTargetType(platform::CPUPlace()), TargetType::kHost);
  ASSERT_EQ(GetLiteTargetType(platform::CUDAPlace(0)), TargetType::kCUDA);
}

TEST(LiteEngineOp, GetLitePrecisionType) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  PrecisionType GetLitePrecisionType(framework::proto::VarType::Type type);
  ASSERT_EQ(GetLitePrecisionType(framework::proto::VarType_Type_FP32),
            PrecisionType::kFloat);
  ASSERT_EQ(GetLitePrecisionType(framework::proto::VarType_Type_INT8),
            PrecisionType::kInt8);
  ASSERT_EQ(GetLitePrecisionType(framework::proto::VarType_Type_INT32),
            PrecisionType::kInt32);
  EXPECT_ANY_THROW(
      GetLitePrecisionType(framework::proto::VarType_Type_SELECTED_ROWS));
}

TEST(LiteEngineOp, GetNativePrecisionType) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  framework::proto::VarType::Type GetNativePrecisionType(
      const PrecisionType& type);
  ASSERT_EQ(GetNativePrecisionType(PrecisionType::kFloat),
            framework::proto::VarType_Type_FP32);
  ASSERT_EQ(GetNativePrecisionType(PrecisionType::kInt8),
            framework::proto::VarType_Type_INT8);
  ASSERT_EQ(GetNativePrecisionType(PrecisionType::kInt32),
            framework::proto::VarType_Type_INT32);
  EXPECT_ANY_THROW(GetNativePrecisionType(PrecisionType::kUnk));
}

TEST(LiteEngineOp, GetNativeLayoutType) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  phi::DataLayout GetNativeLayoutType(const DataLayoutType& type);
  ASSERT_EQ(GetNativeLayoutType(DataLayoutType::kNCHW), phi::DataLayout::kNCHW);
  EXPECT_ANY_THROW(GetNativeLayoutType(DataLayoutType::kNHWC));
}

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
  platform::CPUPlace place;
  phi::CPUContext ctx(place);
  // Prepare variables.
  std::vector<std::string> repetitive_params{"x", "y"};
  CreateTensor(&scope, "x", std::vector<int64_t>({2, 4}));
  CreateTensor(&scope, "y", std::vector<int64_t>({2, 4}));
  ASSERT_EQ(block_->ops_size(), 4);
  *model = program.Proto()->SerializeAsString();
  serialize_params(param, &scope, repetitive_params);
}

template <typename T>
void test_lite_tensor_data_ptr(PrecisionType precision_type) {
  void* GetLiteTensorDataPtr(paddle::lite_api::Tensor * src,
                             PrecisionType precision_type,
                             TargetType target_type);
  std::vector<T> lite_tensor_data({0, 1, 2, 3, 4, 5, 6, 7});
  inference::lite::EngineConfig config;
  make_fake_model(&(config.model), &(config.param));
  LOG(INFO) << "prepare config";
  const std::string unique_key("engine_0");
  config.model_from_memory = true;
  config.valid_places = {
#if defined(PADDLE_WITH_ARM)
    paddle::lite_api::Place({TARGET(kARM), PRECISION(kFloat)}),
#else
    paddle::lite_api::Place({TARGET(kX86), PRECISION(kFloat)}),
#endif
    paddle::lite_api::Place({TARGET(kHost), PRECISION(kAny)}),
  };

  LOG(INFO) << "Create EngineManager";
  inference::Singleton<inference::lite::EngineManager>::Global().Create(
      unique_key, config);
  paddle::lite_api::PaddlePredictor* engine_0 =
      inference::Singleton<inference::lite::EngineManager>::Global().Get(
          unique_key);
  CHECK_NOTNULL(engine_0);
  auto lite_api_tensor = engine_0->GetInput(0);
  lite_api_tensor->Resize(
      std::vector<int64_t>({static_cast<int>(lite_tensor_data.size())}));
  lite_api_tensor->CopyFromCpu(lite_tensor_data.data());
  T* data = static_cast<T*>(GetLiteTensorDataPtr(
      lite_api_tensor.get(), precision_type, TargetType::kHost));
  for (size_t i = 0; i < 8; ++i) {
    CHECK_EQ(data[i], static_cast<T>(i)) << "the i-th num is not correct.";
  }
}

TEST(LiteEngineOp, GetLiteTensorDataPtr) {
  test_lite_tensor_data_ptr<float>(PrecisionType::kFloat);
  test_lite_tensor_data_ptr<int32_t>(PrecisionType::kInt32);
  test_lite_tensor_data_ptr<int8_t>(PrecisionType::kInt8);
  EXPECT_ANY_THROW(test_lite_tensor_data_ptr<float>(PrecisionType::kUnk));
}

void test_tensor_copy(const platform::DeviceContext& ctx) {
  // Create LoDTensor.
  std::vector<float> vector({1, 2, 3, 4});
  phi::DenseTensor lod_tensor;
  framework::TensorFromVector(vector, ctx, &lod_tensor);
  framework::LoD lod({{0, 2, 4}});
  lod_tensor.Resize({4, 1});
  lod_tensor.set_lod(lod);
  // Create lite::Tensor and copy.
  inference::lite::EngineConfig config;
  make_fake_model(&(config.model), &(config.param));
  LOG(INFO) << "prepare config";
  const std::string unique_key("engine_0");
  config.model_from_memory = true;
  config.valid_places = {
#if defined(PADDLE_WITH_ARM)
    paddle::lite_api::Place({TARGET(kARM), PRECISION(kFloat)}),
#else
    paddle::lite_api::Place({TARGET(kX86), PRECISION(kFloat)}),
#endif
    paddle::lite_api::Place({TARGET(kHost), PRECISION(kAny)}),
  };
  LOG(INFO) << "Create EngineManager";
  inference::Singleton<inference::lite::EngineManager>::Global().Create(
      unique_key, config);
  paddle::lite_api::PaddlePredictor* engine_0 =
      inference::Singleton<inference::lite::EngineManager>::Global().Get(
          unique_key);
  CHECK_NOTNULL(engine_0);
  auto lite_api_tensor = engine_0->GetInput(0);
  lite_api_tensor->Resize(
      std::vector<int64_t>({static_cast<int>(vector.size())}));
  lite_api_tensor->CopyFromCpu(vector.data());
  TensorCopyAsync(lite_api_tensor.get(), lod_tensor, ctx);
  // Copy to LoDTensor.
  phi::DenseTensor lod_tensor_n;
  TensorCopyAsync(&lod_tensor_n, *(lite_api_tensor.get()), ctx);
  std::vector<float> result;
  paddle::framework::TensorToVector(lod_tensor_n, ctx, &result);
  ASSERT_EQ(result, vector);
  ASSERT_EQ(lod_tensor_n.lod(), lod_tensor.lod());
}

void test_tensor_share(const platform::DeviceContext& ctx) {
  std::vector<float> vector({1, 2, 3, 4});
  phi::DenseTensor lod_tensor;
  framework::TensorFromVector(vector, ctx, &lod_tensor);
  framework::LoD lod({{0, 2, 4}});
  lod_tensor.Resize({4, 1});
  lod_tensor.set_lod(lod);
  // Create lite::Tensor and share.
  inference::lite::EngineConfig config;
  make_fake_model(&(config.model), &(config.param));
  LOG(INFO) << "prepare config";
  const std::string unique_key("engine_0");
  config.model_from_memory = true;
  config.valid_places = {
#if defined(PADDLE_WITH_ARM)
    paddle::lite_api::Place({TARGET(kARM), PRECISION(kFloat)}),
#else
    paddle::lite_api::Place({TARGET(kX86), PRECISION(kFloat)}),
#endif
    paddle::lite_api::Place({TARGET(kHost), PRECISION(kAny)}),
  };
  LOG(INFO) << "Create EngineManager";
  inference::Singleton<inference::lite::EngineManager>::Global().Create(
      unique_key, config);
  paddle::lite_api::PaddlePredictor* engine_0 =
      inference::Singleton<inference::lite::EngineManager>::Global().Get(
          unique_key);
  CHECK_NOTNULL(engine_0);
  auto lite_api_tensor = engine_0->GetInput(0);
  lite_api_tensor->Resize(
      std::vector<int64_t>({static_cast<int>(vector.size())}));
  lite_api_tensor->CopyFromCpu(vector.data());
  TensorDataShare(lite_api_tensor.get(), &lod_tensor);
  // Copy to LoDTensor.
  phi::DenseTensor lod_tensor_n;
  TensorCopyAsync(&lod_tensor_n, *(lite_api_tensor.get()), ctx);
  std::vector<float> result;
  paddle::framework::TensorToVector(lod_tensor_n, ctx, &result);
  ASSERT_EQ(result, vector);
  ASSERT_EQ(lod_tensor_n.lod(), lod_tensor.lod());
}

TEST(LiteEngineOp, TensorCopyAsync) {
  auto* ctx_cpu =
      platform::DeviceContextPool::Instance().Get(platform::CPUPlace());
  test_tensor_copy(*ctx_cpu);
}

TEST(LiteEngineOp, TensorShare) {
  auto* ctx_cpu =
      platform::DeviceContextPool::Instance().Get(platform::CPUPlace());
  test_tensor_share(*ctx_cpu);
}

}  // namespace utils
}  // namespace lite
}  // namespace inference
}  // namespace paddle
