/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/tests/core/allocator.h"

namespace phi {
namespace tests {

TEST(dense_tensor, meta) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  // TODO(Shixiaowei02): need to check the lod is valid.
  const LoD lod{};

  DenseTensorMeta meta_0;
  CHECK(!meta_0.valid());

  DenseTensorMeta meta_1(dtype, dims);
  CHECK(meta_1.dtype == dtype);
  CHECK(meta_1.dims == dims);
  CHECK(meta_1.valid());

  DenseTensorMeta meta_2(dtype, dims, layout);
  CHECK(meta_2.dtype == dtype);
  CHECK(meta_2.dims == dims);
  CHECK(meta_2.layout == layout);
  CHECK(meta_2.valid());

  DenseTensorMeta meta_3(dtype, dims, layout, lod);
  CHECK(meta_3.dtype == dtype);
  CHECK(meta_3.dims == dims);
  CHECK(meta_3.layout == layout);
  CHECK(meta_3.lod == lod);
  CHECK(meta_3.valid());

  DenseTensorMeta meta_4(meta_3);
  CHECK(meta_4.dtype == dtype);
  CHECK(meta_4.dims == dims);
  CHECK(meta_4.layout == layout);
  CHECK(meta_4.lod == lod);
  CHECK(meta_4.valid());

  DenseTensorMeta meta_5(std::move(meta_4));
  CHECK(meta_5.dtype == dtype);
  CHECK(meta_5.dims == dims);
  CHECK(meta_5.layout == layout);
  CHECK(meta_5.lod == lod);
  CHECK(meta_5.valid());
}

TEST(dense_tensor, def_ctor) {
  DenseTensor tensor_0;
  CHECK(tensor_0.valid());
}

TEST(dense_tensor, ctor) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const LoD lod{};
  DenseTensorMeta meta(dtype, dims, layout, lod);

  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();

  auto check_dense_tensor = [](const DenseTensor& t,
                               const DenseTensorMeta& m) -> bool {
    bool r{true};
    r = r && (t.numel() == product(m.dims));
    r = r && (t.dims() == m.dims);
    r = r && (t.dtype() == m.dtype);
    r = r && (t.layout() == m.layout);
    r = r && (t.place() == phi::CPUPlace());
    r = r && t.initialized();
    r = r && t.IsSharedWith(t);
    return r;
  };

  DenseTensor tensor_0(alloc, meta);
  check_dense_tensor(tensor_0, meta);

  DenseTensor tensor_1(alloc, DenseTensorMeta(meta));
  check_dense_tensor(tensor_0, meta);
}

TEST(dense_tensor, resize) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const LoD lod{};
  DenseTensorMeta meta(dtype, dims, layout, lod);

  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();
  DenseTensor tensor_0(alloc, meta);

  CHECK_EQ(tensor_0.capacity(), 2u);
  tensor_0.ResizeAndAllocate({1, 2, 3});
  CHECK_EQ(tensor_0.capacity(), 6u);
}

TEST(dense_tensor, shallow_copy) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const LoD lod{};
  DenseTensorMeta meta(dtype, dims, layout, lod);

  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();
  DenseTensor tensor_0(alloc, meta);

  DenseTensor tensor_1(tensor_0);
  CHECK(tensor_0.meta() == tensor_1.meta());
}

struct TestStorageProperties
    : public StorageProperties,
      public TypeInfoTraits<StorageProperties, NPUStorageProperties> {
  virtual ~TestStorageProperties() = default;
  static const char* name() { return "TestStorageProperties"; }
};

TEST(dense_tensor, storage_properties) {
  const DataType dtype{DataType::FLOAT32};
  const DDim dims({1, 2});
  DenseTensorMeta meta(dtype, dims);

  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  DenseTensor tensor(fancy_allocator.get(), meta);

  // test no storage properties
  bool caught_exception = false;
  try {
    tensor.storage_properties<NPUStorageProperties>();
  } catch (phi::enforce::EnforceNotMet& error) {
    caught_exception = true;
  }
  EXPECT_TRUE(caught_exception);

  // test custom device storage properties
  EXPECT_FALSE(tensor.storage_properties_initialized());
  auto npu_properties = std::make_unique<NPUStorageProperties>();
  npu_properties->storage_format = 3;
  npu_properties->storage_dims = {1, 1, 1, 1, 16};
  tensor.set_storage_properties(std::move(npu_properties));
  EXPECT_TRUE(tensor.storage_properties_initialized());
  auto get_npu_properties = tensor.storage_properties<NPUStorageProperties>();
  CHECK_EQ(get_npu_properties.storage_format, 3);
  CHECK_EQ(get_npu_properties.storage_dims.size(), 5);

  // test error type storage properties
#ifdef PADDLE_WITH_MKLDNN
  caught_exception = false;
  try {
    tensor.storage_properties<OneDNNStorageProperties>();
  } catch (phi::enforce::EnforceNotMet& error) {
    caught_exception = true;
  }
  EXPECT_TRUE(caught_exception);
#endif

  // test copy storage properties
  auto cp_tensor = tensor;
  auto get_cp_npu_properties =
      cp_tensor.storage_properties<NPUStorageProperties>();
  CHECK_EQ(get_cp_npu_properties.storage_format, 3);
  CHECK_EQ(get_cp_npu_properties.storage_dims.size(), 5);
}

}  // namespace tests
}  // namespace phi
