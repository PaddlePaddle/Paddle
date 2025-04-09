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
#include "test/cpp/phi/core/allocator.h"

namespace phi {
namespace tests {

TEST(dense_tensor, meta) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  // TODO(Shixiaowei02): need to check the lod is valid.
  const LegacyLoD lod{};

  DenseTensorMeta meta_0;
  PADDLE_ENFORCE_EQ(meta_0.valid(),
                    false,
                    common::errors::InvalidArgument(
                        "Fail in default DenseTensorMeta. Expected "
                        "meta_0 to be invalid, but got: %s",
                        meta_0.valid()));

  DenseTensorMeta meta_1(dtype, dims);
  PADDLE_ENFORCE_EQ(
      meta_1.dtype,
      dtype,
      common::errors::InvalidArgument("Fail in DenseTensorMeta with dtype and "
                                      "dims. Expected dtype: %s, but got: %s",
                                      dtype,
                                      meta_1.dtype));
  PADDLE_ENFORCE_EQ(
      meta_1.dims,
      dims,
      common::errors::InvalidArgument("Fail in DenseTensorMeta with dtype and "
                                      "dims. Expected dims: %s, but got: %s",
                                      dims,
                                      meta_1.dims));
  PADDLE_ENFORCE_EQ(meta_1.valid(),
                    true,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype and dims. Expected "
                        "meta_1 to be valid, but got: %s",
                        meta_1.valid()));

  DenseTensorMeta meta_2(dtype, dims, layout);
  PADDLE_ENFORCE_EQ(meta_2.dtype,
                    dtype,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims and layout. "
                        "Expected dtype: %s, but got: %s",
                        dtype,
                        meta_2.dtype));
  PADDLE_ENFORCE_EQ(meta_2.dims,
                    dims,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims "
                        "and layout. Expected dims: %s, but got: %s",
                        dims,
                        meta_2.dims));
  PADDLE_ENFORCE_EQ(meta_2.layout,
                    layout,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims and layout. "
                        "Expected layout: %s, but got: %s",
                        layout,
                        meta_2.layout));
  PADDLE_ENFORCE_EQ(meta_2.valid(),
                    true,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims and layout. "
                        "Expected meta_2 to be valid, but got: %s",
                        meta_2.valid()));

  DenseTensorMeta meta_3(dtype, dims, layout, lod);
  PADDLE_ENFORCE_EQ(meta_3.dtype,
                    dtype,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims, layout and "
                        "lod. Expected dtype: %s, but got: %s",
                        dtype,
                        meta_3.dtype));
  PADDLE_ENFORCE_EQ(meta_3.dims,
                    dims,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims, layout and "
                        "lod. Expected dims: %s, but got: %s",
                        dims,
                        meta_3.dims));
  PADDLE_ENFORCE_EQ(meta_3.layout,
                    layout,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims, layout and "
                        "lod. Expected layout: %s, but got: %s",
                        layout,
                        meta_3.layout));
  PADDLE_ENFORCE_EQ(meta_3.legacy_lod,
                    lod,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims, layout and "
                        "lod. Expected lod: %s, but got: %s",
                        lod,
                        meta_3.legacy_lod));
  PADDLE_ENFORCE_EQ(meta_3.valid(),
                    true,
                    common::errors::InvalidArgument(
                        "Fail in DenseTensorMeta with dtype, dims, layout and "
                        "lod. Expected meta_3 to be valid, but got: %s",
                        meta_3.valid()));

  DenseTensorMeta meta_4(meta_3);
  PADDLE_ENFORCE_EQ(
      meta_4.dtype,
      dtype,
      common::errors::InvalidArgument(
          "Fail in copy DenseTensorMeta. Expected dtype: %s, but got: %s",
          dtype,
          meta_4.dtype));
  PADDLE_ENFORCE_EQ(
      meta_4.dims,
      dims,
      common::errors::InvalidArgument(
          "Fail in copy DenseTensorMeta. Expected dims: %s, but got: %s",
          dims,
          meta_4.dims));
  PADDLE_ENFORCE_EQ(
      meta_4.layout,
      layout,
      common::errors::InvalidArgument(
          "Fail in copy DenseTensorMeta. Expected layout: %s, but got: %s",
          layout,
          meta_4.layout));
  PADDLE_ENFORCE_EQ(
      meta_4.legacy_lod,
      lod,
      common::errors::InvalidArgument(
          "Fail in copy DenseTensorMeta. Expected lod: %s, but got: %s",
          lod,
          meta_4.legacy_lod));
  PADDLE_ENFORCE_EQ(
      meta_4.valid(),
      true,
      common::errors::InvalidArgument("Fail in copy DenseTensorMeta. Expected "
                                      "meta_4 to be valid, but got: %s",
                                      meta_4.valid()));

  DenseTensorMeta meta_5(meta_4);
  PADDLE_ENFORCE_EQ(
      meta_5.dtype,
      dtype,
      common::errors::InvalidArgument(
          "Fail in copy DenseTensorMeta. Expected dtype: %s, but got: %s",
          dtype,
          meta_5.dtype));
  PADDLE_ENFORCE_EQ(
      meta_5.dims,
      dims,
      common::errors::InvalidArgument(
          "Fail in copy DenseTensorMeta. Expected dims: %s, but got: %s",
          dims,
          meta_5.dims));
  PADDLE_ENFORCE_EQ(
      meta_5.layout,
      layout,
      common::errors::InvalidArgument(
          "Fail in copy DenseTensorMeta. Expected layout: %s, but got: %s",
          layout,
          meta_5.layout));
  PADDLE_ENFORCE_EQ(
      meta_5.legacy_lod,
      lod,
      common::errors::InvalidArgument(
          "Fail in copy DenseTensorMeta. Expected lod: %s, but got: %s",
          lod,
          meta_5.legacy_lod));
  PADDLE_ENFORCE_EQ(
      meta_5.valid(),
      true,
      common::errors::InvalidArgument("Fail in copy DenseTensorMeta. Expected "
                                      "meta_5 to be valid, but got: %s",
                                      meta_5.valid()));
}

TEST(dense_tensor, def_ctor) {
  DenseTensor tensor_0;
  PADDLE_ENFORCE_EQ(
      tensor_0.valid(),
      true,
      common::errors::InvalidArgument("Fail in default DenseTensor. Expected "
                                      "tensor_0 to be valid, but got: %s",
                                      tensor_0.valid()));
}

TEST(dense_tensor, ctor) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const LegacyLoD lod{};
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
  const LegacyLoD lod{};
  DenseTensorMeta meta(dtype, dims, layout, lod);

  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();
  DenseTensor tensor_0(alloc, meta);

  PADDLE_ENFORCE_EQ(
      tensor_0.capacity(),
      2u,
      common::errors::InvalidArgument(
          "Fail to initialize DenseTensor. Expected capacity: 2, but got: %s",
          tensor_0.capacity()));
  tensor_0.ResizeAndAllocate({1, 2, 3});
  PADDLE_ENFORCE_EQ(
      tensor_0.capacity(),
      6u,
      common::errors::InvalidArgument(
          "Fail to resize DenseTensor. Expected capacity: 6, but got: %s",
          tensor_0.capacity()));
}

TEST(dense_tensor, shallow_copy) {
  const DDim dims({1, 2});
  const DataType dtype{DataType::INT8};
  const DataLayout layout{DataLayout::NHWC};
  const LegacyLoD lod{};
  DenseTensorMeta meta(dtype, dims, layout, lod);

  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();
  DenseTensor tensor_0(alloc, meta);

  DenseTensor tensor_1(tensor_0);
  PADDLE_ENFORCE_EQ(tensor_0.meta(),
                    tensor_1.meta(),
                    common::errors::InvalidArgument(
                        "Fail to copy DenseTensor. Expected tensor_0 and "
                        "tensor_1 to have the same meta"));
}

TEST(dense_tensor, storage_properties) {
  const DataType dtype{DataType::FLOAT32};
  const DDim dims({1, 2});
  DenseTensorMeta meta(dtype, dims);

  auto fancy_allocator = std::unique_ptr<Allocator>(new FancyAllocator);
  DenseTensor tensor(fancy_allocator.get(), meta);

  // test error type storage properties
#ifdef PADDLE_WITH_DNNL
  bool caught_exception = false;
  try {
    tensor.storage_properties<OneDNNStorageProperties>();
  } catch (common::enforce::EnforceNotMet& error) {
    caught_exception = true;
  }
  PADDLE_ENFORCE_EQ(caught_exception,
                    true,
                    common::errors::InvalidArgument(
                        "Fail to get storage properties. Expected an exception "
                        "to be thrown for OneDNNStorageProperties"));
#endif
}

}  // namespace tests
}  // namespace phi
