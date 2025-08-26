// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/phi/kernels/funcs/indexing.h"
#include <limits>
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/funcs/strided_utils.h"
#include "paddle/phi/kernels/index_put_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

inline bool CheckIsDimsMatchBool(const DDim& first, const DDim& second) {
  int ignore_axis1 = 0, ignore_axis2 = 0;
  for (; ignore_axis1 < first.size(); ++ignore_axis1) {
    if (first[ignore_axis1] != 1) {
      break;
    }
  }
  for (; ignore_axis2 < second.size(); ++ignore_axis2) {
    if (second[ignore_axis2] != 1) {
      break;
    }
  }

  if (second.size() == ignore_axis2) {
    // second tensor has only one value
    return true;
  }

  if (first.size() - ignore_axis1 >= second.size() - ignore_axis2) {
    auto idx1 = first.size() - 1;
    auto idx2 = second.size() - 1;
    bool is_match = true;
    for (; idx2 >= ignore_axis2; idx2--) {
      if (first[idx1--] != second[idx2] && second[idx2] != 1) {
        is_match = false;
        break;
      }
    }
    if (is_match) {
      return true;
    }
  }

  return false;
}

template <typename Context>
phi::DenseTensor Tensor2Contiguous(const Context& dev_ctx,
                                   const phi::DenseTensor& tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  PD_VISIT_ALL_TYPES(tensor.dtype(), "Tensor2Contiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

template <typename T, typename Context>
void LaunchIndexPutKernel_V2(const Context& dev_ctx,
                             const DenseTensor& x,
                             const std::vector<const DenseTensor*>& indices,
                             const DenseTensor& value,
                             bool accumulate,
                             DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      value.dtype(),
      common::errors::InvalidArgument(
          "The data type of tensor value must be same to the data type "
          "of tensor x."));
  PADDLE_ENFORCE_EQ(
      indices.empty(),
      false,
      common::errors::InvalidArgument("Indices cannot be empty."));

  funcs::AdvancedIndex ad =
      funcs::AdvancedIndex<T, Context>(dev_ctx, x, indices);
  if (!CheckIsDimsMatchBool(ad.src.dims(), value.dims())) {
    phi::IndexPutKernel<T, Context>(
        dev_ctx, x, indices, value, accumulate, out);
    return;
  }

  int64_t numel = 0;
  int64_t num_indices = ad.indexed_sizes.size();

  DenseTensorIteratorConfig config;
  config.add_output(ad.src);
  config.add_const_input(value);
  for (size_t i = 0; i < ad.indices.size(); i++) {
    config.add_const_input(*(ad.indices[i]));
  }
  DenseTensorIterator iter = config.build();

  auto sizes = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  auto index_ptrs = std::array<const char*, phi::DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = ad.indexed_sizes[i];
    strides[i] = ad.indexed_strides[i];
    index_ptrs[i] = reinterpret_cast<const char*>(iter.data_ptr(i + 2));
  }

  funcs::OffsetCalculator offset_calc = funcs::make_offset_calculator<3>(iter);

  const int64_t N = iter.numel();
  PADDLE_ENFORCE(N >= 0 && N <= std::numeric_limits<int32_t>::max(),
                 "N >= 0 && N <= std::numeric_limits<int32_t>::max()");
  constexpr int nt = 128;
  constexpr int vt = 4;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = dev_ctx.stream();

  auto* val_data = value.data<T>();

  bool is_initialized = out->initialized();
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (!is_initialized) {
    StridedTensorCopy<T>(x,
                         common::vectorize<int64_t>(x.dims()),
                         common::vectorize<int64_t>(x.strides()),
                         x.offset(),
                         out);
  }

  const char* in_ptr = reinterpret_cast<const char*>(val_data);
  char* out_ptr = reinterpret_cast<char*>(out_data);
  funcs::index_put_kernel<nt, vt, T><<<grid, block, 0, stream>>>(
      N, accumulate, [=] __device__(int idx, bool accumulate) {
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0];
        const char* const in_data = in_ptr + offsets[1];

        int64_t offset = 0;
#pragma unroll
        for (int64_t i = 0; i < num_indices; i++) {
          int64_t index =
              *reinterpret_cast<const int64_t*>(index_ptrs[i] + offsets[2]);
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }
        if (accumulate) {
          *reinterpret_cast<T*>(out_data + offset) +=
              *reinterpret_cast<const T*>(in_data);
        } else {
          *reinterpret_cast<T*>(out_data + offset) =
              *reinterpret_cast<const T*>(in_data);
        }
      });
}

template <typename T, typename Context>
void IndexPutKernel_V2(const Context& dev_ctx,
                       const DenseTensor& x,
                       const std::vector<const DenseTensor*>& indices,
                       const DenseTensor& value,
                       bool accumulate,
                       DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  DenseTensor x_;
  DenseTensor value_;
  for (size_t i = 0; i < indices.size(); i++) {
    PADDLE_ENFORCE_EQ(indices[i]->meta().is_contiguous(),
                      true,
                      common::errors::InvalidArgument(
                          "Indices in Index_put must be contiguous."));
  }

  if (!FLAGS_use_stride_compute_kernel || x.offset() != 0 ||
      value.offset() != 0) {
    if (!x.meta().is_contiguous() || x.offset() != 0) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
    if (!value.meta().is_contiguous() || value.offset() != 0) {
      value_ = Tensor2Contiguous<Context>(dev_ctx, value);
    } else {
      value_ = value;
    }
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::IndexPutKernel<T, Context>(
        dev_ctx, x_, indices, value_, accumulate, out);
    return;
  }
  x_ = x;
  value_ = value;
  if (!FLAGS_use_stride_compute_kernel) {
    PADDLE_THROW(
        common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
                              "Kernel using DenseTensorIterator "
                              "be called, something wrong has happened!"));
  }
  LaunchIndexPutKernel_V2<T, Context>(
      dev_ctx, x_, indices, value_, accumulate, out);
}

}  // namespace phi

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(index_put,
                   GPU,
                   STRIDED,
                   phi::IndexPutKernel_V2,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#endif
