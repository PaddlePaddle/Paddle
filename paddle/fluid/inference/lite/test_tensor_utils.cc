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
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/lite/tensor_utils.h"

namespace paddle {
namespace inference {
namespace lite {
namespace utils {

using paddle::lite_api::TargetType;
using paddle::lite_api::PrecisionType;
using paddle::lite_api::DataLayoutType;

TEST(LiteEngineOp, GetNativePlace) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  platform::Place GetNativePlace(const TargetType& type, int id = 0);
  EXPECT_TRUE(platform::is_cpu_place(GetNativePlace(TargetType::kHost)));
  EXPECT_TRUE(platform::is_gpu_place(GetNativePlace(TargetType::kCUDA)));
  ASSERT_DEATH(GetNativePlace(TargetType::kUnk), "");
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
  ASSERT_DEATH(
      GetLitePrecisionType(framework::proto::VarType_Type_SELECTED_ROWS), "");
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
  ASSERT_DEATH(GetNativePrecisionType(PrecisionType::kUnk), "");
}

TEST(LiteEngineOp, GetNativeLayoutType) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  framework::DataLayout GetNativeLayoutType(const DataLayoutType& type);
  ASSERT_EQ(GetNativeLayoutType(DataLayoutType::kNCHW),
            framework::DataLayout::kNCHW);
  ASSERT_DEATH(GetNativeLayoutType(DataLayoutType::kNHWC), "");
}

void test_tensor_copy(const platform::DeviceContext& ctx) {
  // Create LoDTensor.
  std::vector<float> vector({1, 2, 3, 4});
  framework::LoDTensor lod_tensor;
  framework::TensorFromVector(vector, &lod_tensor);
  framework::LoD lod({{0, 2, 4}});
  lod_tensor.Resize({4, 1});
  lod_tensor.set_lod(lod);
  // Create lite::Tensor and copy.
  paddle::lite::Tensor lite_tensor;
  TensorCopyAsync(&lite_tensor, lod_tensor, ctx);
  // Copy to LoDTensor.
  framework::LoDTensor lod_tensor_n;
  TensorCopyAsync(&lod_tensor_n, lite_tensor, ctx);
#ifdef PADDLE_WITH_CUDA
  if (platform::is_gpu_place(ctx.GetPlace())) {
    platform::GpuStreamSync(
        static_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
  std::vector<float> result;
  TensorToVector(lod_tensor_n, &result);
  ASSERT_EQ(result, vector);
  ASSERT_EQ(lod_tensor_n.lod(), lod_tensor.lod());
}

TEST(LiteEngineOp, TensorCopyAsync) {
  auto* ctx_cpu =
      platform::DeviceContextPool::Instance().Get(platform::CPUPlace());
  test_tensor_copy(*ctx_cpu);
#ifdef PADDLE_WITH_CUDA
  auto* ctx_gpu =
      platform::DeviceContextPool::Instance().Get(platform::CUDAPlace(0));
  test_tensor_copy(*ctx_gpu);
#endif
}

}  // namespace utils
}  // namespace lite
}  // namespace inference
}  // namespace paddle
