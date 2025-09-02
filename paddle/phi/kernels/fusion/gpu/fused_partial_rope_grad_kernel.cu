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

#include "paddle/phi/kernels/fusion/gpu/fused_partial_rope_utils.h"

namespace phi {
namespace fusion {

using FastDivMod = phi::funcs::FastDivMod<uint32_t>;

template <typename T, int VecSize, int NopeSize, int PeSize>
__global__ void rope_grad_kernel(const T* __restrict__ cos,
                                 const T* __restrict__ sin,
                                 const T* __restrict__ out_grad,
                                 T* __restrict__ x_grad,
                                 FastDivMod seq_len,
                                 FastDivMod num_heads,
                                 uint32_t nope_head_dim,
                                 uint32_t pe_head_dim,
                                 uint32_t block_num) {
  using VT = phi::kps::details::VectorType<T, VecSize>;
  extern __shared__ T shm[];

  const uint32_t block_idx = blockIdx.x * 8 + threadIdx.y;
  if (block_idx >= block_num) return;
  const uint32_t seq_idx = seq_len.Divmod(num_heads.Div(block_idx))[1];
  const size_t block_offset =
      static_cast<size_t>(block_idx) * (nope_head_dim + pe_head_dim);
  T* const pe_buffer = shm + threadIdx.y * pe_head_dim;

  // copy nope part
  LOOP_WITH_SIZE_HINT(
      i, threadIdx.x * VecSize, nope_head_dim, 32 * VecSize, NopeSize) {
    size_t idx = block_offset + i;
    *reinterpret_cast<VT*>(x_grad + idx) =
        *reinterpret_cast<const VT*>(out_grad + idx);
  }

  // load pe part, apply embedding and transpose in shared memory
  LOOP_WITH_SIZE_HINT(
      i, threadIdx.x * VecSize, pe_head_dim, 32 * VecSize, PeSize) {
    VT grad = *reinterpret_cast<const VT*>(out_grad + block_offset +
                                           nope_head_dim + i);
    VT grad_rot;
    if (i < pe_head_dim / 2) {
      grad_rot = *reinterpret_cast<const VT*>(
          out_grad + block_offset + nope_head_dim + (i + pe_head_dim / 2));
    } else {
      grad_rot = *reinterpret_cast<const VT*>(
          out_grad + block_offset + nope_head_dim + (i - pe_head_dim / 2));
    }

    VT cos_v = *reinterpret_cast<const VT*>(cos + seq_idx * pe_head_dim + i);
    VT sin_v;
    if (i < pe_head_dim / 2) {
      sin_v = *reinterpret_cast<const VT*>(sin + seq_idx * pe_head_dim +
                                           (i + pe_head_dim / 2));
    } else {
      sin_v = *reinterpret_cast<const VT*>(sin + seq_idx * pe_head_dim +
                                           (i - pe_head_dim / 2));
    }

    for (uint32_t j = 0; j < VecSize; j++) {
      uint32_t pe_idx = i + j;
      if (pe_idx < pe_head_dim / 2) {
        pe_buffer[pe_idx * 2] =
            grad.val[j] * cos_v.val[j] + grad_rot.val[j] * sin_v.val[j];
      } else {
        pe_buffer[(pe_idx - pe_head_dim / 2) * 2 + 1] =
            grad.val[j] * cos_v.val[j] - grad_rot.val[j] * sin_v.val[j];
      }
    }
  }
#ifdef PADDLE_WITH_HIP
  __syncthreads();
#else
  __syncwarp();
#endif

  // store
  LOOP_WITH_SIZE_HINT(
      i, threadIdx.x * VecSize, pe_head_dim, 32 * VecSize, PeSize) {
    VT tmp;
    for (uint32_t j = 0; j < VecSize; j++) {
      tmp.val[j] = pe_buffer[i + j];
    }
    *reinterpret_cast<VT*>(x_grad + block_offset + nope_head_dim + i) = tmp;
  }
}

template <typename T, typename Context>
void FusedPartialRoPEGradKernel(const Context& dev_ctx,
                                const DenseTensor& cos,
                                const DenseTensor& sin,
                                const DenseTensor& out_grad,
                                DenseTensor* x_grad) {
  const auto x_dims = out_grad.dims();
  const int64_t batch_size = x_dims[0];
  const int64_t seq_len = x_dims[1];
  const int64_t num_heads = x_dims[2];
  const int64_t head_dim = x_dims[3];
  const int64_t pe_head_dim = cos.dims()[3];
  const int64_t nope_head_dim = head_dim - pe_head_dim;

  // Allocate x_grad
  dev_ctx.template Alloc<T>(x_grad);

  if (batch_size == 0 || seq_len == 0 || num_heads == 0 || head_dim == 0) {
    return;
  }

  // Launch kernel
  int64_t block_num = batch_size * seq_len * num_heads;
  dim3 grid((block_num + 7) / 8);
  dim3 block(32, 8);
  int64_t shm_size = block.y * pe_head_dim * sizeof(T);

  auto kernel = [&]() {
    SWITCH_ROPE_KERNEL(nope_head_dim, pe_head_dim, {
      return rope_grad_kernel<T, VecSize, NopeSize, PeSize>;
    });
  }();

  kernel<<<grid, block, shm_size, dev_ctx.stream()>>>(
      cos.data<T>(),
      sin.data<T>(),
      out_grad.data<T>(),
      x_grad->data<T>(),
      static_cast<uint32_t>(seq_len),
      static_cast<uint32_t>(num_heads),
      static_cast<uint32_t>(nope_head_dim),
      static_cast<uint32_t>(pe_head_dim),
      static_cast<uint32_t>(block_num));
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_partial_rope_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedPartialRoPEGradKernel,
                   phi::dtype::bfloat16) {}
