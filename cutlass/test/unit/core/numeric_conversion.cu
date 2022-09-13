/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Unit tests for conversion operators.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/numeric_conversion.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace core {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conversion template
template <typename Destination, typename Source, int Count>
__global__ void convert(
  cutlass::Array<Destination, Count> *destination, 
  cutlass::Array<Source, Count> const *source) {

  cutlass::NumericArrayConverter<Destination, Source, Count> convert;

  *destination = convert(*source);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace core
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f32_to_f16_rn) {

  int const kN = 1;
  using Source = float;
  using Destination = cutlass::half_t;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> destination({1, kN});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> source({1, kN});

  for (int i = 0; i < kN; ++i) {
    source.host_data()[i] = float(i);
  }

  source.sync_device();

  test::core::kernel::convert<Destination, Source, 1><<< grid, block >>>(
    reinterpret_cast<cutlass::Array<Destination, 1> *>(destination.device_data()),
    reinterpret_cast<cutlass::Array<Source, 1> const *>(source.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(float(destination.host_data()[i]) == source.host_data()[i]);
  }
}

TEST(NumericConversion, f32x8_to_f16x8_rn) {

  int const kN = 8;
  using Source = float;
  using Destination = cutlass::half_t;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  cutlass::HostTensor<Destination, cutlass::layout::RowMajor> destination({1, kN});
  cutlass::HostTensor<Source, cutlass::layout::RowMajor> source({1, kN});

  for (int i = 0; i < kN; ++i) {
    source.host_data()[i] = float(i);
  }

  source.sync_device();

  test::core::kernel::convert<Destination, Source, kN><<< grid, block >>>(
    reinterpret_cast<cutlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<cutlass::Array<Source, kN> const *>(source.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(float(destination.host_data()[i]) == source.host_data()[i]);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(NumericConversion, f16_to_f32_rn) {
  
  int const kN = 1;
  using Source = cutlass::half_t;
  using Destination = float;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  cutlass::HostTensor<float, cutlass::layout::RowMajor> destination({1, kN});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> source({1, kN});

  for (int i = 0; i < kN; ++i) {
    source.host_data()[i] = Source(i);
  }

  source.sync_device();

  test::core::kernel::convert<Destination, Source, kN><<< grid, block >>>(
    reinterpret_cast<cutlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<cutlass::Array<Source, kN> const *>(source.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(float(destination.host_data()[i]) == float(source.host_data()[i]));
  }
}

TEST(NumericConversion, f16x8_to_f32x8_rn) {

  int const kN = 8;
  using Source = cutlass::half_t;
  using Destination = float;

  dim3 grid(1, 1);
  dim3 block(1, 1);

  cutlass::HostTensor<float, cutlass::layout::RowMajor> destination({1, kN});
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> source({1, kN});

  for (int i = 0; i < kN; ++i) {
    source.host_data()[i] = float(i);
  }

  source.sync_device();

  test::core::kernel::convert<Destination, Source, kN><<< grid, block >>>(
    reinterpret_cast<cutlass::Array<Destination, kN> *>(destination.device_data()),
    reinterpret_cast<cutlass::Array<Source, kN> const *>(source.device_data())
  );

  destination.sync_host();

  for (int i = 0; i < kN; ++i) {
    EXPECT_TRUE(float(destination.host_data()[i]) == float(source.host_data()[i]));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
