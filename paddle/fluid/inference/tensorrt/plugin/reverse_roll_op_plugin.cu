// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

/*******************  invokeReverseRoll  ***********************/
// src is [batch*window_num, window_len, dim]
// dst is [batch, H, W, dim] + rolled
// grid(W, H, batch)
// block(min(1024, dim))

template<typename T>
__global__ void reverse_roll(T*        dst,
                             const T*  src,
                             const int batch,
                             const int window_num,
                             const int window_len,
                             const int window_size,
                             const int H,
                             const int W,
                             const int shift_size,
                             const int dim)
{
    const int batch_idx     = blockIdx.z;
    const int H_idx_shifted = (blockIdx.y + shift_size) % H;
    const int W_idx_shifted = (blockIdx.x + shift_size) % W;
    const int H_idx         = blockIdx.y;
    const int W_idx         = blockIdx.x;
    const int window_idx    = H_idx / window_size * (W / window_size) + W_idx / window_size;
    const int idx_in_window = (H_idx % window_size) * window_size + (W_idx % window_size);
    const int input_offset  = (batch_idx * window_num + window_idx) * window_len + idx_in_window;
    const int output_offset = (batch_idx * H + H_idx_shifted) * W + W_idx_shifted;
    for (int tid = threadIdx.x; tid < dim; tid += blockDim.x) {
        dst[output_offset * dim + tid] = src[input_offset * dim + tid];
    }
}

// src is [batch*window_num, window_len, dim]
// dst is [batch, H, W, dim] + rolled
// grid(W, H, batch)
// block(min(1024, dim))
template<typename T>
void invokeReverseRoll(T*           dst,
                       const T*     src,
                       int          batch,
                       int          window_num,
                       int          window_len,
                       int          window_size,
                       int          H,
                       int          W,
                       int          dim,
                       int          shift_size,
                       cudaStream_t stream)
{
    dim3 grid(W, H, batch);
    int  blockSize = dim;
#ifdef ENABLE_BF16
    if ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && (dim % 2 == 0)) {
#else
    if (std::is_same<T, half>::value && (dim % 2 == 0)) {
#endif
        blockSize = dim / 2;
        if (blockSize > 1024) {
            blockSize = 1024;
        }
        using T2 = typename TypeConverter<T>::Type;  // bfloat162 or half2
        reverse_roll<<<grid, blockSize, 0, stream>>>(
            (T2*)dst, (const T2*)src, batch, window_num, window_len, window_size, H, W, shift_size, dim / 2);
    }
    else {
        if (blockSize > 1024) {
            blockSize = 1024;
        }
        reverse_roll<<<grid, blockSize, 0, stream>>>(
            dst, src, batch, window_num, window_len, window_size, H, W, shift_size, dim);
    }
}

template void invokeReverseRoll(float*       dst,
                                const float* src,
                                int          batch,
                                int          window_num,
                                int          window_len,
                                int          window_size,
                                int          H,
                                int          W,
                                int          dim,
                                int          shift_size,
                                cudaStream_t stream);

template void invokeReverseRoll(half*        dst,
                                const half*  src,
                                int          batch,
                                int          window_num,
                                int          window_len,
                                int          window_size,
                                int          H,
                                int          W,
                                int          dim,
                                int          shift_size,
                                cudaStream_t stream);


} // plugin
} // tensorrt
} // inference
} // paddle
