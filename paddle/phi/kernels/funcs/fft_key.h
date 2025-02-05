// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/fft.h"

namespace phi {
namespace funcs {
namespace detail {

const int64_t kMaxFFTNdim = 3;
const int64_t kMaxDataNdim = kMaxFFTNdim + 1;

struct FFTConfigKey {
  int signal_ndim_;  // 1 <= signal_ndim <= kMaxFFTNdim
  // These include additional batch dimension as well.
  int64_t sizes_[kMaxDataNdim];
  int64_t input_shape_[kMaxDataNdim];
  int64_t output_shape_[kMaxDataNdim];
  FFTTransformType fft_type_;
  DataType value_type_;

  using shape_t = std::vector<int64_t>;
  FFTConfigKey() = default;

  FFTConfigKey(const shape_t& in_shape,
               const shape_t& out_shape,
               const shape_t& signal_size,
               FFTTransformType fft_type,
               DataType value_type) {
    // Padding bits must be zeroed for hashing
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_size.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;
    std::copy(signal_size.cbegin(), signal_size.cend(), sizes_);
    std::copy(in_shape.cbegin(), in_shape.cend(), input_shape_);
    std::copy(out_shape.cbegin(), out_shape.cend(), output_shape_);
  }
};

// Hashing machinery for Key
// Fowler–Noll–Vo hash function
// see
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Key>
struct KeyHash {
  // Key must be a POD because we read out its memory
  // contents as char* when hashing
  static_assert(std::is_pod<Key>::value, "Key must be plain old data type");

  size_t operator()(const Key& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < static_cast<int>(sizeof(Key)); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return static_cast<size_t>(value);
  }
};

template <typename Key>
struct KeyEqual {
  // Key must be a POD because we read out its memory
  // contents as char* when comparing
  static_assert(std::is_pod<Key>::value, "Key must be plain old data type");

  bool operator()(const Key& a, const Key& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(Key)) == 0;
  }
};

static FFTConfigKey create_fft_configkey(const DenseTensor& input,
                                         const DenseTensor& output,
                                         int signal_ndim) {
  // Create the transform plan (either from cache or locally)
  DataType input_dtype = input.dtype();
  const auto value_type =
      IsComplexType(input_dtype) ? ToRealType(input_dtype) : input_dtype;
  const auto fft_type = GetFFTTransformType(input.dtype(), output.dtype());
  // signal sizes
  std::vector<int64_t> signal_size(signal_ndim + 1);

  signal_size[0] = input.dims()[0];
  for (int64_t i = 1; i <= signal_ndim; ++i) {
    auto in_size = input.dims()[i];
    auto out_size = output.dims()[i];
    signal_size[i] = std::max(in_size, out_size);
  }
  FFTConfigKey key(common::vectorize(input.dims()),
                   common::vectorize(output.dims()),
                   signal_size,
                   fft_type,
                   value_type);
  return key;
}

}  // namespace detail
}  // namespace funcs
}  // namespace phi
