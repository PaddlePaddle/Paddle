// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/TensorOptions.h>

#include <optional>

#include "gtest/gtest.h"

// ============================================================
// Tests for c10::TensorOptions
// ============================================================

namespace {

class DefaultDtypeGuard {
 public:
  explicit DefaultDtypeGuard(c10::ScalarType dtype)
      : previous_(c10::get_default_dtype()) {
    c10::set_default_dtype(c10::scalarTypeToTypeMeta(dtype));
  }

  ~DefaultDtypeGuard() { c10::set_default_dtype(previous_); }

 private:
  caffe2::TypeMeta previous_;
};

}  // namespace

// Default-constructed TensorOptions has no device/dtype/layout/grad fields set
TEST(TensorOptionsTest, DefaultConstructor_NothingSet) {
  c10::TensorOptions opts;

  ASSERT_FALSE(opts.has_device());
  ASSERT_FALSE(opts.has_dtype());
  ASSERT_FALSE(opts.has_layout());
  ASSERT_FALSE(opts.has_requires_grad());
  ASSERT_FALSE(opts.has_pinned_memory());
  ASSERT_FALSE(opts.has_memory_format());
}

// Default dtype falls back to Float, device to CPU, layout to kStrided
TEST(TensorOptionsTest, DefaultConstructor_Defaults) {
  c10::TensorOptions opts;

  ASSERT_EQ(opts.dtype(), c10::ScalarType::Float);
  ASSERT_EQ(opts.device(), c10::Device(c10::kCPU));
  ASSERT_EQ(opts.layout(), c10::kStrided);
  ASSERT_FALSE(opts.requires_grad());
  ASSERT_FALSE(opts.pinned_memory());
}

// ---- dtype ----

TEST(TensorOptionsTest, SetDtype_HasDtypeTrue) {
  c10::TensorOptions opts = c10::TensorOptions().dtype(c10::kFloat);

  ASSERT_TRUE(opts.has_dtype());
  ASSERT_EQ(opts.dtype(), c10::kFloat);
}

TEST(TensorOptionsTest, SetDtype_Double) {
  c10::TensorOptions opts = c10::TensorOptions().dtype(c10::kDouble);

  ASSERT_EQ(opts.dtype(), c10::kDouble);
}

TEST(TensorOptionsTest, SetDtype_Int32) {
  c10::TensorOptions opts = c10::TensorOptions().dtype(c10::kInt);

  ASSERT_EQ(opts.dtype(), c10::kInt);
}

TEST(TensorOptionsTest, SetDtype_Bool) {
  c10::TensorOptions opts = c10::TensorOptions().dtype(c10::kBool);

  ASSERT_EQ(opts.dtype(), c10::kBool);
}

// Implicit construction from ScalarType
TEST(TensorOptionsTest, ImplicitFromScalarType) {
  c10::TensorOptions opts(c10::kHalf);

  ASSERT_TRUE(opts.has_dtype());
  ASSERT_EQ(opts.dtype(), c10::kHalf);
}

TEST(TensorOptionsTest, ImplicitFromTypeMeta) {
  c10::TensorOptions opts(caffe2::TypeMeta::Make<double>());

  ASSERT_TRUE(opts.has_dtype());
  ASSERT_EQ(opts.dtype(), caffe2::TypeMeta::Make<double>());
}

TEST(TensorOptionsTest, SetDtype_TypeMetaOptional) {
  c10::TensorOptions opts = c10::TensorOptions().dtype(
      std::make_optional(caffe2::TypeMeta::Make<int>()));

  ASSERT_TRUE(opts.has_dtype());
  ASSERT_EQ(opts.dtype(), caffe2::TypeMeta::Make<int>());
}

TEST(TensorOptionsTest, SetDtype_TypeMetaTemplateMember) {
  c10::TensorOptions opts;
  opts.dtype<int64_t>();

  ASSERT_TRUE(opts.has_dtype());
  ASSERT_EQ(opts.dtype(), caffe2::TypeMeta::Make<int64_t>());
}

TEST(TensorOptionsTest, ClearDtypeWithNullopt) {
  c10::TensorOptions opts = c10::TensorOptions().dtype(
      std::make_optional(caffe2::TypeMeta::Make<float>()));
  opts = opts.dtype(std::optional<caffe2::TypeMeta>{});

  ASSERT_FALSE(opts.has_dtype());
  ASSERT_FALSE(opts.dtype_opt().has_value());
}

TEST(TensorOptionsTest, DtypeOptReturnsTypeMeta) {
  c10::TensorOptions opts =
      c10::TensorOptions().dtype(caffe2::TypeMeta::Make<bool>());

  ASSERT_TRUE(opts.dtype_opt().has_value());
  ASSERT_EQ(opts.dtype_opt().value(), caffe2::TypeMeta::Make<bool>());
}

TEST(TensorOptionsTest, DtypeDefaultTracksGlobalDefaultDtype) {
  DefaultDtypeGuard guard(c10::kDouble);
  c10::TensorOptions opts;

  ASSERT_EQ(opts.dtype(), caffe2::TypeMeta::Make<double>());
  ASSERT_EQ(c10::dtype_or_default(std::optional<caffe2::TypeMeta>{}),
            caffe2::TypeMeta::Make<double>());
  ASSERT_EQ(c10::dtype_or_default(std::optional<c10::ScalarType>{}),
            c10::kDouble);
}

TEST(TensorOptionsTest, DefaultComplexDtypeTracksGlobalDefaultDtype) {
  {
    DefaultDtypeGuard guard(c10::kHalf);

    ASSERT_EQ(c10::get_default_dtype_as_scalartype(), c10::kHalf);
    ASSERT_EQ(c10::get_default_complex_dtype().toScalarType(),
              c10::ScalarType::ComplexHalf);
  }

  {
    DefaultDtypeGuard guard(c10::kDouble);

    ASSERT_EQ(c10::get_default_dtype_as_scalartype(), c10::kDouble);
    ASSERT_EQ(c10::get_default_complex_dtype().toScalarType(),
              c10::ScalarType::ComplexDouble);
  }

  {
    DefaultDtypeGuard guard(c10::kFloat);

    ASSERT_EQ(c10::get_default_dtype_as_scalartype(), c10::kFloat);
    ASSERT_EQ(c10::get_default_complex_dtype().toScalarType(),
              c10::ScalarType::ComplexFloat);
  }
}

// ---- device ----

TEST(TensorOptionsTest, SetDevice_CPU) {
  c10::TensorOptions opts = c10::TensorOptions().device(c10::Device(c10::kCPU));

  ASSERT_TRUE(opts.has_device());
  ASSERT_EQ(opts.device().type(), c10::DeviceType::CPU);
}

TEST(TensorOptionsTest, SetDevice_CUDA) {
  c10::TensorOptions opts =
      c10::TensorOptions().device(c10::Device(c10::kCUDA, 0));

  ASSERT_TRUE(opts.has_device());
  ASSERT_EQ(opts.device().type(), c10::DeviceType::CUDA);
  ASSERT_EQ(opts.device().index(), 0);
}

// Helper function: c10::device()
TEST(TensorOptionsTest, HelperFunction_device) {
  c10::TensorOptions opts = c10::device(c10::Device(c10::kCPU));

  ASSERT_TRUE(opts.has_device());
  ASSERT_EQ(opts.device().type(), c10::DeviceType::CPU);
}

// ---- layout ----

TEST(TensorOptionsTest, SetLayout_kStrided) {
  c10::TensorOptions opts = c10::TensorOptions().layout(c10::kStrided);

  ASSERT_TRUE(opts.has_layout());
  ASSERT_EQ(opts.layout(), c10::kStrided);
}

TEST(TensorOptionsTest, SetLayout_kSparse) {
  c10::TensorOptions opts = c10::TensorOptions().layout(c10::kSparse);

  ASSERT_TRUE(opts.has_layout());
  ASSERT_EQ(opts.layout(), c10::kSparse);
}

TEST(TensorOptionsTest, SetLayout_kSparseCsr) {
  c10::TensorOptions opts = c10::TensorOptions().layout(c10::kSparseCsr);

  ASSERT_TRUE(opts.has_layout());
  ASSERT_EQ(opts.layout(), c10::kSparseCsr);
}

// Implicit construction from Layout
TEST(TensorOptionsTest, ImplicitFromLayout) {
  c10::TensorOptions opts(c10::kSparse);

  ASSERT_TRUE(opts.has_layout());
  ASSERT_EQ(opts.layout(), c10::kSparse);
}

// Helper function: c10::layout()
TEST(TensorOptionsTest, HelperFunction_layout) {
  c10::TensorOptions opts = c10::layout(c10::kSparse);

  ASSERT_TRUE(opts.has_layout());
  ASSERT_EQ(opts.layout(), c10::kSparse);
}

// ---- is_sparse / is_sparse_csr / is_sparse_compressed ----

TEST(TensorOptionsTest, IsSparse_kSparse) {
  c10::TensorOptions opts = c10::TensorOptions().layout(c10::kSparse);

  ASSERT_TRUE(opts.is_sparse());
  ASSERT_FALSE(opts.is_sparse_csr());
  ASSERT_FALSE(opts.is_sparse_compressed());
}

TEST(TensorOptionsTest, IsSparse_kSparseCsr) {
  c10::TensorOptions opts = c10::TensorOptions().layout(c10::kSparseCsr);

  ASSERT_FALSE(opts.is_sparse());
  ASSERT_TRUE(opts.is_sparse_csr());
  ASSERT_TRUE(opts.is_sparse_compressed());
}

TEST(TensorOptionsTest, IsSparse_kSparseCsc) {
  c10::TensorOptions opts = c10::TensorOptions().layout(c10::kSparseCsc);

  ASSERT_FALSE(opts.is_sparse());
  ASSERT_FALSE(opts.is_sparse_csr());
  ASSERT_TRUE(opts.is_sparse_compressed());
}

TEST(TensorOptionsTest, IsSparse_kStrided) {
  c10::TensorOptions opts = c10::TensorOptions().layout(c10::kStrided);

  ASSERT_FALSE(opts.is_sparse());
  ASSERT_FALSE(opts.is_sparse_csr());
  ASSERT_FALSE(opts.is_sparse_compressed());
}

// ---- requires_grad ----

TEST(TensorOptionsTest, SetRequiresGrad_True) {
  c10::TensorOptions opts = c10::TensorOptions().requires_grad(true);

  ASSERT_TRUE(opts.has_requires_grad());
  ASSERT_TRUE(opts.requires_grad());
}

TEST(TensorOptionsTest, SetRequiresGrad_False) {
  c10::TensorOptions opts = c10::TensorOptions().requires_grad(false);

  ASSERT_TRUE(opts.has_requires_grad());
  ASSERT_FALSE(opts.requires_grad());
}

// Helper function: c10::requires_grad()
TEST(TensorOptionsTest, HelperFunction_requires_grad) {
  c10::TensorOptions opts = c10::requires_grad(true);

  ASSERT_TRUE(opts.has_requires_grad());
  ASSERT_TRUE(opts.requires_grad());
}

// ---- pinned_memory ----

TEST(TensorOptionsTest, SetPinnedMemory_True) {
  c10::TensorOptions opts = c10::TensorOptions().pinned_memory(true);

  ASSERT_TRUE(opts.has_pinned_memory());
  ASSERT_TRUE(opts.pinned_memory());
}

TEST(TensorOptionsTest, SetPinnedMemory_False) {
  c10::TensorOptions opts = c10::TensorOptions().pinned_memory(false);

  ASSERT_TRUE(opts.has_pinned_memory());
  ASSERT_FALSE(opts.pinned_memory());
}

// ---- memory_format ----

TEST(TensorOptionsTest, SetMemoryFormat_Contiguous) {
  c10::TensorOptions opts =
      c10::TensorOptions().memory_format(c10::MemoryFormat::Contiguous);

  ASSERT_TRUE(opts.has_memory_format());
  ASSERT_EQ(opts.memory_format_opt().value(), c10::MemoryFormat::Contiguous);
}

TEST(TensorOptionsTest, SetMemoryFormat_ChannelsLast) {
  c10::TensorOptions opts =
      c10::TensorOptions().memory_format(c10::MemoryFormat::ChannelsLast);

  ASSERT_TRUE(opts.has_memory_format());
  ASSERT_EQ(opts.memory_format_opt().value(), c10::MemoryFormat::ChannelsLast);
}

// Helper function: c10::memory_format()
TEST(TensorOptionsTest, HelperFunction_memory_format) {
  c10::TensorOptions opts = c10::memory_format(c10::MemoryFormat::Contiguous);

  ASSERT_TRUE(opts.has_memory_format());
}

// ---- merge_memory_format ----

TEST(TensorOptionsTest, MergeMemoryFormat_Overrides) {
  c10::TensorOptions base =
      c10::TensorOptions().memory_format(c10::MemoryFormat::Contiguous);
  c10::TensorOptions merged =
      base.merge_memory_format(c10::MemoryFormat::ChannelsLast);

  ASSERT_EQ(merged.memory_format_opt().value(),
            c10::MemoryFormat::ChannelsLast);
}

TEST(TensorOptionsTest, MergeMemoryFormat_NulloptKeepsOriginal) {
  c10::TensorOptions base =
      c10::TensorOptions().memory_format(c10::MemoryFormat::Contiguous);
  c10::TensorOptions merged = base.merge_memory_format(std::nullopt);

  ASSERT_EQ(merged.memory_format_opt().value(), c10::MemoryFormat::Contiguous);
}

// ---- chaining multiple options ----

TEST(TensorOptionsTest, ChainMultipleOptions) {
  c10::TensorOptions opts = c10::TensorOptions()
                                .dtype(c10::kDouble)
                                .device(c10::Device(c10::kCPU))
                                .layout(c10::kStrided)
                                .requires_grad(true);

  ASSERT_EQ(opts.dtype(), c10::kDouble);
  ASSERT_EQ(opts.device().type(), c10::DeviceType::CPU);
  ASSERT_EQ(opts.layout(), c10::kStrided);
  ASSERT_TRUE(opts.requires_grad());
}

// ---- opt() accessor returns nullopt when not set ----

TEST(TensorOptionsTest, DeviceOpt_NulloptWhenNotSet) {
  c10::TensorOptions opts;
  ASSERT_FALSE(opts.device_opt().has_value());
}

TEST(TensorOptionsTest, DtypeOpt_NulloptWhenNotSet) {
  c10::TensorOptions opts;
  ASSERT_FALSE(opts.dtype_opt().has_value());
}

TEST(TensorOptionsTest, LayoutOpt_NulloptWhenNotSet) {
  c10::TensorOptions opts;
  ASSERT_FALSE(opts.layout_opt().has_value());
}

TEST(TensorOptionsTest, RequiresGradOpt_NulloptWhenNotSet) {
  c10::TensorOptions opts;
  ASSERT_FALSE(opts.requires_grad_opt().has_value());
}

TEST(TensorOptionsTest, PinnedMemoryOpt_NulloptWhenNotSet) {
  c10::TensorOptions opts;
  ASSERT_FALSE(opts.pinned_memory_opt().has_value());
}

TEST(TensorOptionsTest, MemoryFormatOpt_NulloptWhenNotSet) {
  c10::TensorOptions opts;
  ASSERT_FALSE(opts.memory_format_opt().has_value());
}

// ---- Implicit construction from MemoryFormat ----

TEST(TensorOptionsTest, ImplicitFromMemoryFormat) {
  c10::TensorOptions opts(c10::MemoryFormat::ChannelsLast);

  ASSERT_TRUE(opts.has_memory_format());
  ASSERT_EQ(opts.memory_format_opt().value(), c10::MemoryFormat::ChannelsLast);
}

// ---- Helper free functions ----

TEST(TensorOptionsTest, HelperFunction_dtype) {
  c10::TensorOptions opts = c10::dtype(c10::kLong);

  ASSERT_TRUE(opts.has_dtype());
  ASSERT_EQ(opts.dtype(), c10::kLong);
}

TEST(TensorOptionsTest, HelperFunction_dtypeTypeMeta) {
  c10::TensorOptions opts = c10::dtype(caffe2::TypeMeta::Make<int16_t>());

  ASSERT_TRUE(opts.has_dtype());
  ASSERT_EQ(opts.dtype(), caffe2::TypeMeta::Make<int16_t>());
}

TEST(TensorOptionsTest, HelperFunction_dtypeTemplate) {
  c10::TensorOptions opts = c10::dtype<uint8_t>();

  ASSERT_TRUE(opts.has_dtype());
  ASSERT_EQ(opts.dtype(), caffe2::TypeMeta::Make<uint8_t>());
  ASSERT_EQ(c10::typeMetaToScalarType(opts.dtype()), c10::kByte);
}
