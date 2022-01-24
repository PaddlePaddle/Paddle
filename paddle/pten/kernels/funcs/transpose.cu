// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/funcs/transpose.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace pten {

namespace math {

#define REINTERPRET(T, DST_PTR, SRC_PTR) \
  T* DST_PTR = reinterpret_cast<T*>(SRC_PTR)

template <typename T>
__global__ void TransposeNormalKernel(const T* in_ptr,
                                      T* out_ptr,
                                      int64_t element,
                                      const int64_t* in_stride_ptr,
                                      const int64_t* out_stride_ptr,
                                      const int64_t* axis_ptr,
                                      int rank) {
  CUDA_KERNEL_LOOP(out_idx, element) {
    int64_t in_idx = 0;
    int64_t tmp_idx = out_idx;
    for (int i = 0; i < rank; ++i) {
      const int64_t coordinate = tmp_idx / out_stride_ptr[i];
      tmp_idx -= coordinate * out_stride_ptr[i];
      in_idx += coordinate * in_stride_ptr[axis_ptr[i]];
    }
    out_ptr[out_idx] = in_ptr[in_idx];
  }
}

template <typename T>
struct TransposeNormal<GPUContext, T> {
  // for dims >= 7 situation
  void operator()(const GPUContext& dev_ctx,
                  const pten::DenseTensor& in,
                  pten::DenseTensor* out,
                  const std::vector<int64_t>& axis) {
    const int rank = axis.size();
    auto in_stride = paddle::framework::stride(in.dims());
    auto out_stride = paddle::framework::stride(out->dims());
    auto* in_ptr = in.data<T>();
    auto* out_ptr = out->mutable_data<T>();

    // copy in_stride, out_stride, axis to gpu device
    const paddle::platform::CUDAPlace& cuda_place = dev_ctx.GetPlace();
    paddle::platform::CPUPlace cpu_place = paddle::platform::CPUPlace();
    size_t size = 3 * rank * sizeof(int64_t);
    auto cpu_buf_holder = paddle::memory::Alloc(cpu_place, size);
    auto cuda_buf_holder = paddle::memory::Alloc(cuda_place, size);
    REINTERPRET(int64_t, cpu_buf, cpu_buf_holder->ptr());
    REINTERPRET(int64_t, cuda_buf, cuda_buf_holder->ptr());
    for (int i = 0; i < rank; ++i) {
      cpu_buf[i] = in_stride[i];
      cpu_buf[rank + i] = out_stride[i];
      cpu_buf[2 * rank + i] = axis[i];
    }
    paddle::memory::Copy(
        cuda_place, cuda_buf, cpu_place, cpu_buf, size, dev_ctx.stream());
    REINTERPRET(const int64_t, in_stride_ptr, cuda_buf);
    REINTERPRET(const int64_t, out_stride_ptr, cuda_buf + rank);
    REINTERPRET(const int64_t, axis_ptr, cuda_buf + 2 * rank);

    const int MAX_BLOCK_DIM = dev_ctx.GetMaxThreadsPerBlock();
    const int MAX_GRID_DIM =
        dev_ctx.GetMaxPhysicalThreadCount() / MAX_BLOCK_DIM;
    int64_t elements = in.numel();
    int block_size = (elements >= MAX_BLOCK_DIM)
                         ? MAX_BLOCK_DIM
                         : (1 << static_cast<int>(std::log2(elements)));
    int grid_size = elements / block_size;
    grid_size = (grid_size >= MAX_GRID_DIM) ? MAX_GRID_DIM : grid_size;
    TransposeNormalKernel<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
        in_ptr,
        out_ptr,
        elements,
        in_stride_ptr,
        out_stride_ptr,
        axis_ptr,
        rank);
  }
};

// define transpose normal
#define DEFINE_GPU_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<GPUContext, TYPE>

DEFINE_GPU_TRANS_NORMAL(bool);
DEFINE_GPU_TRANS_NORMAL(int8_t);
DEFINE_GPU_TRANS_NORMAL(uint8_t);
DEFINE_GPU_TRANS_NORMAL(int16_t);
DEFINE_GPU_TRANS_NORMAL(uint16_t);
DEFINE_GPU_TRANS_NORMAL(int32_t);
DEFINE_GPU_TRANS_NORMAL(uint32_t);
DEFINE_GPU_TRANS_NORMAL(int64_t);
DEFINE_GPU_TRANS_NORMAL(uint64_t);
DEFINE_GPU_TRANS_NORMAL(float);
DEFINE_GPU_TRANS_NORMAL(double);
DEFINE_GPU_TRANS_NORMAL(paddle::platform::float16);
DEFINE_GPU_TRANS_NORMAL(paddle::platform::bfloat16);
DEFINE_GPU_TRANS_NORMAL(paddle::platform::complex<float>);
DEFINE_GPU_TRANS_NORMAL(paddle::platform::complex<double>);

}  // namespace math
}  // namespace pten
