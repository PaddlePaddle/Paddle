/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/contiguous_kernel.h"

#include <set>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/transpose_grad_kernel_impl.h"

#if defined(PADDLE_WITH_OPENMP)
#include <omp.h>
#endif

namespace phi {

inline int64_t DivUp(const int64_t& x, const int64_t& y) {
  return (x + y - 1) / y;
}

inline void DealWithStride(const DenseTensorIterator& iter, int64_t* strides) {
  for (int dim = 0; dim < iter.ndim(); dim++) {
    for (int arg = 0; arg < iter.ntensors(); arg++) {
      *strides++ = iter.strides(arg)[dim];
    }
  }
  if (iter.ndim() < 2) {
    std::fill_n(strides, (2 - iter.ndim()) * iter.ntensors(), 0);
  }
}

template <typename T>
inline void FallbackContiguous(const DDim& input_dims,
                               const DDim& input_stride,
                               const int64_t numel,
                               const T* input_data,
                               T* output_data) {
  int rank = input_dims.size();
  auto dims = input_dims;
  for (int64_t i = 0; i < numel; i++) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t mod = index_tmp % dims[dim];
      index_tmp = index_tmp / dims[dim];
      input_offset += mod * input_stride[dim];
    }
    output_data[i] = input_data[input_offset];
  }
}

inline bool OnlyTransposed(const DDim& shape,
                           const DDim& stride,
                           const uint64_t& offset) {
  if (offset != 0) {
    return false;
  }

  DDim x_stride = stride;
  DDim x_shape = shape;

  std::set<int> visited_idx;
  for (int i = 0; i < stride.size(); i++) {
    int64_t max_num = 0;
    int max_idx = -1;
    for (int j = 0; j < stride.size(); j++) {
      if (visited_idx.count(j)) {
        continue;
      }
      if (stride[j] < 1) {
        return false;
      }
      if (stride[j] > max_num) {
        max_num = stride[j];
        max_idx = j;
      }
    }
    if (max_idx == -1) {
      return false;
    }
    if (i != 0 && x_stride[i - 1] == max_num) {
      return false;
    }
    visited_idx.insert(max_idx);
    x_stride[i] = max_num;
    x_shape[i] = shape[max_idx];
  }

  if (DenseTensorMeta::calc_strides(x_shape) == x_stride) {
    return true;
  } else {
    return false;
  }
}

inline bool FastContiguousJudge(const std::vector<int64_t>& coalesce_shape,
                                const DenseTensor& input) {
  if (coalesce_shape.size() > 3 || input.dims().size() == 0) return false;
  auto x_meta = input.meta();
  if (coalesce_shape.size() == 3 &&
      !OnlyTransposed(x_meta.dims, x_meta.strides, x_meta.offset))
    return false;
  return true;
}

inline bool FastTransposeCopyValid(const DenseTensor& self,
                                   const DenseTensor& src) {
  constexpr int64_t MIN_NUMEL = 360;
  return src.numel() != 0 && src.dims().size() == 2 && src.strides()[0] == 1 &&
         src.strides()[1] == src.dims()[0] &&
         self.dims().size() == src.dims().size() && self.numel() >= MIN_NUMEL;
}

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);

  const T* input_data = input.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  auto numel = input.numel();

  if (numel == 0) {
    return;
  }

  if (IsComplexType(input.dtype())) {
    FallbackContiguous<T>(
        input.dims(), input.strides(), numel, input_data, output_data);
    return;
  }

#if defined(_WIN32)
  FallbackContiguous<T>(
      input.dims(), input.strides(), numel, input_data, output_data);
  return;
#else
  if (FastTransposeCopyValid(*out, input)) {
    constexpr int64_t TRANS_NUMEL = 60;
    void* trans_buffer =
        malloc(phi::SizeOf(input.dtype()) * TRANS_NUMEL * TRANS_NUMEL);

    const T* tmp_src_ptr = input_data;
    T* tmp_out_ptr = output_data;
    T* tmp_buf_ptr = reinterpret_cast<T*>(trans_buffer);

    int64_t dim0 = out->dims()[0];
    int64_t dim1 = out->dims()[1];

    for (int64_t d0 = 0; d0 < dim0; d0 += TRANS_NUMEL) {
      for (int64_t d1 = 0; d1 < dim1; d1 += TRANS_NUMEL) {
        const T* src_ptr_inter = tmp_src_ptr + d0 + d1 * dim0;
        T* out_ptr_inter = tmp_out_ptr + d1 + d0 * dim1;

        int nr = std::min(dim0 - d0, TRANS_NUMEL);
        int nc = std::min(dim1 - d1, TRANS_NUMEL);

        for (int c = 0; c < nc; c++) {
          memcpy(tmp_buf_ptr + c * TRANS_NUMEL,
                 src_ptr_inter + c * dim0,
                 nr * sizeof(T));
        }

        int rc_max = std::max(nr, nc);
        int rc_min = std::min(nr, nc);
        for (int r = 0; r < rc_max; r++) {
          int end = std::min(r, rc_min);
          for (int c = 0; c < end; c++) {
            T tmp = tmp_buf_ptr[r + TRANS_NUMEL * c];
            tmp_buf_ptr[r + TRANS_NUMEL * c] = tmp_buf_ptr[r * TRANS_NUMEL + c];
            tmp_buf_ptr[r * TRANS_NUMEL + c] = tmp;
          }
        }

        for (int r = 0; r < nr; r++) {
          memcpy(out_ptr_inter + r * dim1,
                 tmp_buf_ptr + r * TRANS_NUMEL,
                 nc * sizeof(T));
        }
      }
    }
    free(trans_buffer);
  } else {
#if defined(PADDLE_WITH_OPENMP)
    phi::DenseTensorIteratorConfig config;
    config.add_output(*out);
    config.add_const_input(input);
    config.is_alloc_out_ = true;
    phi::DenseTensorIterator iter = config.build();
    if (!FastContiguousJudge(iter.shape(), input)) {
      FallbackContiguous<T>(
          input.dims(), input.strides(), numel, input_data, output_data);
      return;
    }

    std::vector<int64_t> tmp_strides(
        iter.ntensors() * static_cast<size_t>(std::max(iter.ndim(), 2)));

    DealWithStride(iter, tmp_strides.data());

    std::vector<int64_t> out_stride(tmp_strides.begin() + iter.ntensors(),
                                    tmp_strides.end());

    const int64_t& iter_numel = iter.numel();
    const char* in_ptr = reinterpret_cast<const char*>(input_data);
    char* out_ptr = reinterpret_cast<char*>(output_data);
    int64_t end = iter_numel;
    int64_t begin = 0;
    int64_t grain_size = 32768;

    int64_t* whole_stride = tmp_strides.data();

#pragma omp parallel
    {
      int64_t num_threads = omp_get_num_threads();

      if (grain_size > 0) {
        num_threads = std::min(num_threads, DivUp((end - begin), grain_size));
      }

      int64_t tid = omp_get_thread_num();
      int64_t chunk_size = DivUp((end - begin), num_threads);
      int64_t begin_tid = begin + tid * chunk_size;

      if (begin_tid < end) {
        int64_t range_start = begin_tid;
        int64_t range_end = std::min(end, chunk_size + begin_tid);

        auto dimiter = DimIter(iter.shape(), range_start, range_end);
        while (!dimiter.iter_to_end()) {
          const auto v_ndim = dimiter.values.size();
          const char* tmp_in_data = in_ptr;
          char* tmp_out_data = out_ptr;
          for (size_t dim = 0; dim < v_ndim; dim++) {
            int64_t value = dimiter.values[dim];
            tmp_out_data += value * whole_stride[dim * iter.ntensors() + 0];
            tmp_in_data += value * whole_stride[dim * iter.ntensors() + 1];
          }

          auto step = dimiter.iter_for_step();

          for (int64_t i = 0; i < step[1]; i++) {
            for (int64_t j = 0; j < step[0]; j++) {
              const char* real_in_ptr = tmp_in_data + j * whole_stride[1];
              char* real_out_ptr = tmp_out_data + j * whole_stride[0];

              *reinterpret_cast<T*>(real_out_ptr) =
                  *reinterpret_cast<const T*>(real_in_ptr);
            }
            tmp_in_data = tmp_in_data + out_stride[1];
            tmp_out_data = tmp_out_data + out_stride[0];
          }

          dimiter.iter_to_next(step);
        }
      }
    }
#else
    FallbackContiguous<T>(
        input.dims(), input.strides(), numel, input_data, output_data);
#endif
  }
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(contiguous,
                   CPU,
                   ALL_LAYOUT,
                   phi::ContiguousKernel,
                   bool,
                   uint8_t,
                   uint16_t,
                   uint32_t,
                   uint64_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2) {}
