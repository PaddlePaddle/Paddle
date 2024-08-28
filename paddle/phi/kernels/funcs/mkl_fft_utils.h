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
#include <functional>
#include <numeric>
#include "paddle/phi/backends/dynload/mklrt.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/kernels/funcs/fft.h"

namespace phi {
namespace funcs {
namespace detail {

#define MKL_DFTI_CHECK(expr)                                                 \
  do {                                                                       \
    MKL_LONG status = (expr);                                                \
    if (!phi::dynload::DftiErrorClass(status, DFTI_NO_ERROR))                \
      PADDLE_THROW(                                                          \
          common::errors::External(phi::dynload::DftiErrorMessage(status))); \
  } while (0);

struct DftiDescriptorDeleter {
  void operator()(DFTI_DESCRIPTOR_HANDLE handle) {
    if (handle != nullptr) {
      MKL_DFTI_CHECK(phi::dynload::DftiFreeDescriptor(&handle));
    }
  }
};

// A RAII wrapper for MKL_DESCRIPTOR*
class DftiDescriptor {
 public:
  void init(DFTI_CONFIG_VALUE precision,
            DFTI_CONFIG_VALUE signal_type,
            MKL_LONG signal_ndim,
            MKL_LONG* sizes) {
    PADDLE_ENFORCE_EQ(desc_.get(),
                      nullptr,
                      common::errors::AlreadyExists(
                          "DftiDescriptor has already been initialized."));

    DFTI_DESCRIPTOR* raw_desc;
    MKL_DFTI_CHECK(phi::dynload::DftiCreateDescriptorX(
        &raw_desc, precision, signal_type, signal_ndim, sizes));
    desc_.reset(raw_desc);
  }

  DFTI_DESCRIPTOR* get() const {
    DFTI_DESCRIPTOR* raw_desc = desc_.get();
    PADDLE_ENFORCE_NOT_NULL(raw_desc,
                            common::errors::PreconditionNotMet(
                                "DFTI DESCRIPTOR has not been initialized."));
    return raw_desc;
  }

 private:
  std::unique_ptr<DFTI_DESCRIPTOR, DftiDescriptorDeleter> desc_;
};

static DftiDescriptor plan_mkl_fft(const DataType in_dtype,
                                   const DataType out_dtype,
                                   const phi::DDim& in_strides,
                                   const phi::DDim& out_strides,
                                   const std::vector<int64_t>& signal_sizes,
                                   FFTNormMode normalization,
                                   bool forward) {
  const DFTI_CONFIG_VALUE precision = [&] {
    switch (in_dtype) {
      case DataType::FLOAT32:
        return DFTI_SINGLE;
      case DataType::COMPLEX64:
        return DFTI_SINGLE;
      case DataType::FLOAT64:
        return DFTI_DOUBLE;
      case DataType::COMPLEX128:
        return DFTI_DOUBLE;
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "Invalid input datatype (%s), input data type should be FP32, "
            "FP64, COMPLEX64 or COMPLEX128.",
            in_dtype));
    }
  }();

  // C2C, R2C, C2R
  const FFTTransformType fft_type = GetFFTTransformType(in_dtype, out_dtype);
  const DFTI_CONFIG_VALUE domain =
      (fft_type == FFTTransformType::C2C) ? DFTI_COMPLEX : DFTI_REAL;

  DftiDescriptor descriptor;
  std::vector<MKL_LONG> fft_sizes(signal_sizes.cbegin(), signal_sizes.cend());
  const MKL_LONG signal_ndim = fft_sizes.size() - 1;
  descriptor.init(precision, domain, signal_ndim, fft_sizes.data() + 1);

  // placement inplace or not inplace
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
      descriptor.get(), DFTI_PLACEMENT, DFTI_NOT_INPLACE));

  // number of transformations
  const MKL_LONG batch_size = fft_sizes[0];
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
      descriptor.get(), DFTI_NUMBER_OF_TRANSFORMS, batch_size));

  // input & output distance
  const MKL_LONG idist = in_strides[0];
  const MKL_LONG odist = out_strides[0];
  MKL_DFTI_CHECK(
      phi::dynload::DftiSetValue(descriptor.get(), DFTI_INPUT_DISTANCE, idist));
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
      descriptor.get(), DFTI_OUTPUT_DISTANCE, odist));

  // input & output stride
  std::vector<MKL_LONG> mkl_in_stride(1 + signal_ndim, 0);
  std::vector<MKL_LONG> mkl_out_stride(1 + signal_ndim, 0);
  for (MKL_LONG i = 1; i <= signal_ndim; i++) {
    mkl_in_stride[i] = in_strides[i];
    mkl_out_stride[i] = out_strides[i];
  }
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
      descriptor.get(), DFTI_INPUT_STRIDES, mkl_in_stride.data()));
  MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
      descriptor.get(), DFTI_OUTPUT_STRIDES, mkl_out_stride.data()));

  // conjugate even storage
  if (!(fft_type == FFTTransformType::C2C)) {
    MKL_DFTI_CHECK(phi::dynload::DftiSetValue(
        descriptor.get(), DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
  }

  MKL_LONG signal_numel = std::accumulate(fft_sizes.cbegin() + 1,
                                          fft_sizes.cend(),
                                          1UL,
                                          std::multiplies<MKL_LONG>());
  if (normalization != FFTNormMode::none) {
    const double scale =
        ((normalization == FFTNormMode::by_sqrt_n)
             ? 1.0 / std::sqrt(static_cast<double>(signal_numel))
             : 1.0 / static_cast<double>(signal_numel));
    const auto scale_direction = [&]() {
      if (fft_type == FFTTransformType::R2C ||
          (fft_type == FFTTransformType::C2C && forward)) {
        return DFTI_FORWARD_SCALE;
      } else {
        // (fft_type == FFTTransformType::C2R ||
        //          (fft_type == FFTTransformType::C2C && !forward))
        return DFTI_BACKWARD_SCALE;
      }
    }();
    MKL_DFTI_CHECK(
        phi::dynload::DftiSetValue(descriptor.get(), scale_direction, scale));
  }

  // commit the descriptor
  MKL_DFTI_CHECK(phi::dynload::DftiCommitDescriptor(descriptor.get()));
  return descriptor;
}

}  // namespace detail
}  // namespace funcs
}  // namespace phi
