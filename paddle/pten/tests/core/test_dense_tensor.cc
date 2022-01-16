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

#include "gtest/gtest.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/tests/core/allocator.h"

namespace pten {
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
    r = r && (t.place() == paddle::platform::CPUPlace());
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
  tensor_0.mutable_data<int8_t>();
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

}  // namespace tests
}  // namespace pten
