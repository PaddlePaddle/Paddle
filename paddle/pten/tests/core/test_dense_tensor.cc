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

TEST(dense_tensor, shape) {
  const DDim dims({1, 2});
  const DataLayout layout{DataLayout::NHWC};
  const std::vector<std::vector<size_t>> lod{};

  DenseTensorShape shape_0;
  CHECK(!shape_0.valid());

  DenseTensorShape shape_1(dims);
  CHECK(shape_1.dims == dims);
  CHECK(shape_1.valid());

  DenseTensorShape shape_2(dims, layout);
  CHECK(shape_2.dims == dims);
  CHECK(shape_2.layout == layout);
  CHECK(shape_2.valid());

  DenseTensorShape shape_3(dims, layout, lod);
  CHECK(shape_3.dims == dims);
  CHECK(shape_3.layout == layout);
  CHECK(shape_3.lod == lod);
  CHECK(shape_3.valid());

  DenseTensorShape shape_4(shape_3);
  CHECK(shape_3 == shape_4);

  DenseTensorShape shape_5(std::move(shape_3));
  CHECK(shape_5 == shape_4);
}

TEST(dense_tensor, meta) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  // TODO(Shixiaowei02): need to check the lod is valid.
  const std::vector<std::vector<size_t>> lod{};

  DenseTensorShape shape(dims, layout, lod);
  DenseTensorShape shape_1(shape);

  DenseTensorMeta meta_0;
  CHECK(!meta_0.valid());

  DenseTensorMeta meta_1(dtype, shape);
  CHECK(meta_1.type == dtype);
  CHECK(meta_1.shape == shape);
  CHECK(meta_1.valid());

  DenseTensorMeta meta_2(dtype, std::move(shape));
  CHECK(meta_2.type == dtype);
  CHECK(meta_2.shape == shape_1);
  CHECK(meta_2.valid());

  DenseTensorMeta meta_3(meta_2);
  CHECK(meta_2 == meta_3);

  DenseTensorMeta meta_4(std::move(meta_3));
  CHECK(meta_4 == meta_2);
}

TEST(dense_tensor, def_ctor) {
  DenseTensor tensor_0;
  CHECK(!tensor_0.valid());
}

TEST(dense_tensor, ctor) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const std::vector<std::vector<size_t>> lod{};
  DenseTensorMeta meta(dtype, DenseTensorShape(dims, layout, lod));

  auto alloc = std::make_shared<FancyAllocator>();

  auto check_dense_tensor = [](const DenseTensor& t,
                               const DenseTensorMeta& m) -> bool {
    bool r{true};
    r = r && (t.numel() == product(m.shape.dims));
    r = r && (t.dims() == m.shape.dims);
    r = r && (t.data_type() == m.type);
    r = r && (t.layout() == m.shape.layout);
    r = r && (t.place() == paddle::platform::CPUPlace());
    r = r && t.initialized();
    r = r && t.IsSharedWith(t);
    return r;
  };

  DenseTensor tensor_0(alloc, meta);
  check_dense_tensor(tensor_0, meta);

  DenseTensor tensor_1(alloc, DenseTensorMeta(meta));
  check_dense_tensor(tensor_0, meta);

  DenseTensor tensor_2(make_intrusive<TensorStorage>(alloc), meta);
  CHECK(tensor_2.data<int8_t>() == nullptr);
  CHECK_NOTNULL(tensor_2.mutable_data<int8_t>());
  check_dense_tensor(tensor_2, meta);
}

TEST(dense_tensor, resize) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const std::vector<std::vector<size_t>> lod{};
  DenseTensorMeta meta(dtype, DenseTensorShape(dims, layout, lod));

  auto alloc = std::make_shared<FancyAllocator>();
  DenseTensor tensor_0(alloc, meta);

  CHECK_EQ(tensor_0.memory_size(), 2u);
  tensor_0.check_memory_size();
  tensor_0.Resize({1, 2, 3});
  CHECK_EQ(tensor_0.memory_size(), 0u);
  tensor_0.mutable_data<int8_t>();
  CHECK_EQ(tensor_0.memory_size(), 6u);

  auto shape_0 = tensor_0.shape();
  shape_0.dims = {2, 3};
  tensor_0.set_shape(shape_0);
  CHECK_EQ(tensor_0.memory_size(), 6u);

  auto storage = tensor_0.release();
  CHECK_EQ(storage->size(), 6u);
}

TEST(dense_tensor, shallow_clone) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const std::vector<std::vector<size_t>> lod{};
  DenseTensorMeta meta(dtype, DenseTensorShape(dims, layout, lod));

  auto alloc = std::make_shared<FancyAllocator>();
  DenseTensor tensor_0(alloc, meta);

  auto tensor_1 = tensor_0.shallow_clone();
  CHECK(tensor_0.meta() == tensor_1.meta());
  CHECK(tensor_0.release() == tensor_1.release());
}

}  // namespace tests
}  // namespace pten
