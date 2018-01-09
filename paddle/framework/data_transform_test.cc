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
#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "paddle/framework/data_transform.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {
using namespace platform;

/**
 * @brief cross validation of different kernel type transform
 *  We use four bit map represent different combination.
 *  If the field has multiple possible value, only choose two of them.
 *  For DataType, only test the FP32(float), FP64(double).
 *  e.g. 0000 -> FP32, CPUPlace, kNHWC, kPlain
 *       1111 -> FP64, GPUPlace, kNCHW, kMKLDNN
 */

std::array<proto::DataType, 2> kDataType = {
    {proto::DataType::FP32, proto::DataType::FP64}};

std::array<Place, 2> kPlace = {{CPUPlace(), CUDAPlace(0)}};

std::array<DataLayout, 2> kDataLayout = {{
    DataLayout::kNHWC, DataLayout::kNCHW,
}};

std::array<LibraryType, 2> kLibraryType = {{
    LibraryType::kPlain, LibraryType::kMKLDNN,
}};

OpKernelType GenFromBit(const std::vector<bool> bits) {
  return OpKernelType(kDataType[bits[0]], kPlace[bits[1]], kDataLayout[bits[2]],
                      kLibraryType[bits[3]]);
}

int test_value = 0;

auto kernel0 = GenFromBit({0, 0, 0, 0});
auto kernel1 = GenFromBit({0, 0, 0, 1});
auto kernel2 = GenFromBit({0, 0, 1, 0});
auto kernel3 = GenFromBit({0, 0, 1, 1});

void TransDataType_t(const platform::DeviceContext* ctx,
                     const KernelTypePair& p, const Variable& in,
                     Variable* out) {
  test_value++;
}

void TransDataLayout_t(const platform::DeviceContext* ctx,
                       const KernelTypePair& p, const Variable& in,
                       Variable* out) {
  test_value--;
}

void TransLibraryType_t(const platform::DeviceContext* ctx,
                        const KernelTypePair& p, const Variable& in,
                        Variable* out) {
  test_value += 2;
}

}  // namespace framework
}  // namespace paddle

namespace frw = paddle::framework;

REGISTER_DATA_TRANSFORM_FN(frw::kernel0, frw::kernel1, frw::TransDataType_t);
REGISTER_DATA_TRANSFORM_FN(frw::kernel1, frw::kernel2, frw::TransDataLayout_t);
REGISTER_DATA_TRANSFORM_FN(frw::kernel0, frw::kernel2, frw::TransLibraryType_t);

TEST(DataTransform, Register) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  auto& instance = DataTransformFnMap::Instance();
  paddle::framework::Variable in;
  paddle::framework::Variable out;

  DeviceContext* ctx = new CPUDeviceContext();
  auto pair0 = std::make_pair(frw::kernel0, frw::kernel1);
  instance.Get(pair0)(ctx, pair0, in, &out);
  ASSERT_EQ(test_value, 1);

  auto pair1 = std::make_pair(frw::kernel1, frw::kernel2);
  instance.Get(pair1)(ctx, pair1, in, &out);
  ASSERT_EQ(test_value, 0);

  auto pair3 = std::make_pair(frw::kernel0, frw::kernel2);
  instance.Get(pair3)(ctx, pair3, in, &out);
  ASSERT_EQ(test_value, 2);
}

TEST(DataTransform, DataLayout) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  auto& instance = DataTransformFnMap::Instance();
  Variable in;
  Variable out;
  Tensor* src = in.GetMutable<Tensor>();
  src->mutable_data<double>(make_ddim({2, 3, 1, 2}), CPUPlace());
  src->set_layout(DataLayout::kNHWC);

  DeviceContext* ctx = new CPUDeviceContext();

  {
    auto kernel1 = GenFromBit({1, 0, 0, 0});
    auto kernel2 = GenFromBit({1, 0, 1, 0});
    auto pair0 = std::make_pair(kernel1, kernel2);
    instance.Get(pair0)(ctx, pair0, in, &out);
  }

  Tensor dst = out.Get<Tensor>();

  EXPECT_TRUE(dst.layout() == DataLayout::kNCHW);
  EXPECT_TRUE(dst.dims() == make_ddim({2, 2, 3, 1}));

  {
    auto kernel1 = GenFromBit({1, 0, 1, 0});
    auto kernel2 = GenFromBit({1, 0, 0, 0});
    auto pair0 = std::make_pair(kernel1, kernel2);
    instance.Get(pair0)(ctx, pair0, out, &in);
  }

  EXPECT_TRUE(src->layout() == DataLayout::kNHWC);
  EXPECT_TRUE(src->dims() == make_ddim({2, 3, 1, 2}));
}

TEST(DataTransform, DataType) {
  using namespace paddle::framework;
  using namespace paddle::platform;

  auto& instance = DataTransformFnMap::Instance();
  DeviceContext* ctx = new CPUDeviceContext();

  Variable in;
  Variable out;
  Tensor* src = in.GetMutable<Tensor>();
  float* ptr = src->mutable_data<float>(make_ddim({2, 3}), CPUPlace());
  for (int i = 0; i < 6; ++i) {
    ptr[i] = i / 3;
  }

  {
    auto kernel1 = GenFromBit({0, 0, 0, 0});
    auto kernel2 = GenFromBit({1, 0, 0, 0});
    auto pair0 = std::make_pair(kernel1, kernel2);
    instance.Get(pair0)(ctx, pair0, in, &out);
  }
  Tensor dst = out.Get<Tensor>();
  EXPECT_TRUE(dst.data<double>() != nullptr);
}
