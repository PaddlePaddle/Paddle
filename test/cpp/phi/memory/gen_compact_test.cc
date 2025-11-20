/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"

PD_DECLARE_bool(enable_compact_mem);
PD_DECLARE_int64(max_reserved_threshold_in_gb);
PD_DECLARE_int64(cur_allocated_threshold_in_gb);
PD_DECLARE_bool(try_allocate);
PD_DECLARE_bool(use_multi_scale_virtual_memory_auto_growth);
PD_DECLARE_uint64(vmm_small_pool_size_in_mb);

namespace paddle {
namespace memory {
namespace allocation {
using paddle::experimental::CheckAndDoCompact;
class CheckAndDoCompactTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set default flags
    FLAGS_enable_compact_mem = true;
    FLAGS_try_allocate = true;
    FLAGS_use_multi_scale_virtual_memory_auto_growth = true;
    FLAGS_vmm_small_pool_size_in_mb = 2;
    FLAGS_v = 4;
  }

  void TearDown() override { meta_tensors_.clear(); }

  std::vector<phi::MetaTensor*> meta_tensors_;
};

TEST_F(CheckAndDoCompactTest, DisabledByFlag) {
  FLAGS_enable_compact_mem = false;
  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, NoCompactWhenBelowMaxReservedThreshold) {
  FLAGS_enable_compact_mem = true;
  FLAGS_max_reserved_threshold_in_gb = 80;
  FLAGS_cur_allocated_threshold_in_gb = 0;
  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, NoCompactWhenBelowCurAllocatedThreshold) {
  FLAGS_enable_compact_mem = true;
  FLAGS_max_reserved_threshold_in_gb = 0;
  FLAGS_cur_allocated_threshold_in_gb = 80;
  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, CompactWhenNeeded) {
  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;
}

TEST_F(CheckAndDoCompactTest, SkipZeroNumelTensors) {
  phi::DenseTensor zero_tensor;
  phi::DenseTensorMeta zero_meta(phi::DataType::FLOAT32, phi::DDim({0}));
  zero_tensor.set_meta(zero_meta);
  phi::MetaTensor meta_tensor(zero_tensor);
  meta_tensors_.push_back(&meta_tensor);

  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;

  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, SkipNagetiveNumelTensors) {
  phi::DenseTensor negative_tensor;
  phi::DenseTensorMeta negative_meta(phi::DataType::FLOAT32, phi::DDim({-1}));
  negative_meta.is_scalar = true;
  negative_tensor.set_meta(negative_meta);
  phi::MetaTensor meta_tensor(negative_tensor);
  meta_tensors_.push_back(&meta_tensor);

  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;

  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, ReqLessThenMaxFree) {
  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;
  auto var1 = paddle::experimental::full(
      {10, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());

  var1.reset();

  phi::DenseTensor tensor;
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32, phi::DDim({2, 1024, 1024}));
  tensor.set_meta(meta);
  phi::MetaTensor meta_tensor(tensor);

  meta_tensors_.push_back(&meta_tensor);
  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, ReqMoreThenLargestNFree) {
  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;
  auto var1 = paddle::experimental::full(
      {10, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  var1.reset();

  phi::DenseTensor tensor;
  phi::DenseTensorMeta meta(phi::DataType::FLOAT32,
                            phi::DDim({20, 1024, 1024}));
  tensor.set_meta(meta);
  phi::MetaTensor meta_tensor(tensor);

  meta_tensors_.push_back(&meta_tensor);
  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, TryAllocDisable) {
  FLAGS_try_allocate = false;
  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;
  auto var1 = paddle::experimental::full(
      {10, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto var2 = paddle::experimental::full(
      {2, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto var3 = paddle::experimental::full(
      {5, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  var1.reset();
  var3.reset();

  phi::DenseTensor tensor1;
  phi::DenseTensorMeta meta1(phi::DataType::FLOAT32,
                             phi::DDim({8, 1024, 1024}));
  tensor1.set_meta(meta1);
  phi::MetaTensor meta_tensor1(tensor1);

  phi::DenseTensor tensor2;
  phi::DenseTensorMeta meta2(phi::DataType::FLOAT32,
                             phi::DDim({4, 1024, 1024}));
  tensor2.set_meta(meta2);
  phi::MetaTensor meta_tensor2(tensor2);

  meta_tensors_.push_back(&meta_tensor1);
  meta_tensors_.push_back(&meta_tensor2);
  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, TryAllocSucc) {
  FLAGS_try_allocate = true;
  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;
  auto var1 = paddle::experimental::full(
      {15, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto var2 = paddle::experimental::full(
      {2, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto var3 = paddle::experimental::full(
      {10, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  var1.reset();
  var3.reset();

  phi::DenseTensor tensor1;
  phi::DenseTensorMeta meta1(phi::DataType::FLOAT32,
                             phi::DDim({10, 1024, 1024}));
  tensor1.set_meta(meta1);
  phi::MetaTensor meta_tensor1(tensor1);

  phi::DenseTensor tensor2;
  phi::DenseTensorMeta meta2(phi::DataType::FLOAT32,
                             phi::DDim({9, 1024, 1024}));
  tensor2.set_meta(meta2);
  phi::MetaTensor meta_tensor2(tensor2);

  meta_tensors_.push_back(&meta_tensor1);
  meta_tensors_.push_back(&meta_tensor2);
  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, TryAllocSuccNoSplit) {
  FLAGS_try_allocate = true;
  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;
  auto var1 = paddle::experimental::full(
      {10, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto var2 = paddle::experimental::full(
      {2, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto var3 = paddle::experimental::full(
      {10, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  var1.reset();
  var3.reset();

  phi::DenseTensor tensor1;
  phi::DenseTensorMeta meta1(phi::DataType::FLOAT32,
                             phi::DDim({10, 1024, 1024}));
  tensor1.set_meta(meta1);
  phi::MetaTensor meta_tensor1(tensor1);

  phi::DenseTensor tensor2;
  phi::DenseTensorMeta meta2(phi::DataType::FLOAT32,
                             phi::DDim({10, 1024, 1024}));
  tensor2.set_meta(meta2);
  phi::MetaTensor meta_tensor2(tensor2);

  meta_tensors_.push_back(&meta_tensor1);
  meta_tensors_.push_back(&meta_tensor2);
  CheckAndDoCompact(meta_tensors_, "test_api");
}

TEST_F(CheckAndDoCompactTest, TryAllocFail) {
  FLAGS_try_allocate = true;
  FLAGS_cur_allocated_threshold_in_gb = 0;
  FLAGS_max_reserved_threshold_in_gb = 0;
  auto var1 = paddle::experimental::full(
      {10, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto var2 = paddle::experimental::full(
      {2, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  auto var3 = paddle::experimental::full(
      {5, 1024, 1024}, 1, paddle::DataType::FLOAT32, paddle::GPUPlace());
  var1.reset();
  var3.reset();

  phi::DenseTensor tensor1;
  phi::DenseTensorMeta meta1(phi::DataType::FLOAT32,
                             phi::DDim({11, 1024, 1024}));
  tensor1.set_meta(meta1);
  phi::MetaTensor meta_tensor1(tensor1);

  phi::DenseTensor tensor2;
  phi::DenseTensorMeta meta2(phi::DataType::FLOAT32,
                             phi::DDim({2, 1024, 1024}));
  tensor2.set_meta(meta2);
  phi::MetaTensor meta_tensor2(tensor2);

  meta_tensors_.push_back(&meta_tensor1);
  meta_tensors_.push_back(&meta_tensor2);
  CheckAndDoCompact(meta_tensors_, "test_api");
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
