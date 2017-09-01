/* Copyright (c) 2016 PaddlePaddle Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <iostream>
#include "paddle/framework/op_registry.h"
#include "paddle/platform/assert.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int id) : v_(value), id_(id) {}

  __device__ __forceinline__ void set(T value, int id) {
    v_ = value;
    id_ = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v_ = in.v_;
    id_ = in.id_;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (v_ < value);
  }

  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (v_ < in.v_) || ((v_ == in.v_) && (id_ > in.id_));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (v_ > in.v_) || ((v_ == in.v_) && (id_ < in.id_));
  }

  T v_;
  int id_;
};

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p,
                                      int beam_size) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int beam_size>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* val, int* col,
                                        int idx, int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < val[idx]) {
      Pair<T> tmp(val[idx], col[idx]);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* val, int* col,
                                        int idx, int dim, const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < val[idx]) {
      Pair<T> tmp(val[idx], col[idx]);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[], int& beam,
                                              int beam_size, const T* src,
                                              bool& firstStep, bool& is_empty,
                                              Pair<T>& max, int dim,
                                              const int tid) {
  if (beam > 0) {
    int length = beam < beam_size ? beam : beam_size;
    if (firstStep) {
      firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - beam) {
          topk[k] = topk[k + beam];
        } else {
          topk[k].set(-INFINITY, -1);
        }
      }
      if (!is_empty) {
        GetTopK<T, BlockSize>(topk + MaxLength - beam, src, tid, dim, max,
                              length);
      }
    }

    max = topk[MaxLength - 1];
    if (max.id_ == -1) is_empty = true;
    beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[], int& beam,
                                              int beam_size, const T* val,
                                              int* col, bool& firstStep,
                                              bool& is_empty, Pair<T>& max,
                                              int dim, const int tid) {
  if (beam > 0) {
    int length = beam < beam_size ? beam : beam_size;
    if (firstStep) {
      firstStep = false;
      GetTopK<T, BlockSize>(topk, val, col, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - beam) {
          topk[k] = topk[k + beam];
        } else {
          topk[k].set(-INFINITY, -1);
        }
      }
      if (!is_empty) {
        GetTopK<T, BlockSize>(topk + MaxLength - beam, val, col, tid, dim, max,
                              length);
      }
    }

    max = topk[MaxLength - 1];
    if (max.id_ == -1) is_empty = true;
    beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void BlockReduce(Pair<T>* shTopK, int* maxid,
                                            Pair<T> topk[], T** topVal,
                                            int** topIds, int& beam, int& k,
                                            const int tid, const int warp) {
  while (true) {
    __syncthreads();
    if (tid < BlockSize / 2) {
      if (shTopK[tid] < shTopK[tid + BlockSize / 2]) {
        maxid[tid] = tid + BlockSize / 2;
      } else {
        maxid[tid] = tid;
      }
    }
    __syncthreads();
    for (int stride = BlockSize / 4; stride > 0; stride = stride / 2) {
      if (tid < stride) {
        if (shTopK[maxid[tid]] < shTopK[maxid[tid + stride]]) {
          maxid[tid] = maxid[tid + stride];
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (tid == 0) {
      **topVal = shTopK[maxid[0]].v_;
      **topIds = shTopK[maxid[0]].id_;
      (*topVal)++;
      (*topIds)++;
    }
    if (tid == maxid[0]) beam++;
    if (--k == 0) break;
    __syncthreads();

    if (tid == maxid[0]) {
      if (beam < MaxLength) {
        shTopK[tid] = topk[beam];
      }
    }
    if (maxid[0] / 32 == warp) {
      if (__shfl(beam, (maxid[0]) % 32, 32) == MaxLength) break;
    }
  }
}

/**
 * Each block compute one sample.
 * In a block:
 * 1. every thread get top MaxLength value;
 * 2. merge to shTopK, block reduce and get max value;
 * 3. go to the second setp, until one thread's topk value is null;
 * 4. go to the first setp, until get the topk value.
 */
template <typename T, int MaxLength, int BlockSize>
__global__ void KeMatrixTopK(T* output, int output_stride, int* indices,
                             const T* src, int lds, int dim, int k) {
  __shared__ Pair<T> sh_topk[BlockSize];
  __shared__ int maxid[BlockSize / 2];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;
  output += blockIdx.x * output_stride;
  indices += blockIdx.x * k;

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;

  for (int k = 0; k < MaxLength; k++) {
    topk[k].set(-INFINITY, -1);
  }
  while (k) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk, beam, k,
                                           src + blockIdx.x * lds, firststep,
                                           is_empty, max, dim, tid);

    sh_topk[tid] = topk[0];
    BlockReduce<T, MaxLength, BlockSize>(sh_topk, maxid, topk, &output,
                                         &indices, beam, k, tid, warp);
  }
}

template <typename T, typename AttrType = int>
class TopkOpCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    std::cout << "in TopkOpCUDAKernel" << std::endl;
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");
    size_t k = static_cast<AttrType>(ctx.op_.GetAttr<AttrType>("k"));

    const T* input_data = input->data<T>();

    output->mutable_data<T>(ctx.GetPlace());
    indices->mutable_data<int>(ctx.GetPlace());
    T* output_data = output->data<T>();
    int* indices_data = indices->data<int>();
    size_t input_height = input->dims()[0];
    size_t input_width = input->dims()[1];
    if (k > input_width) k = input_width;

    // pass lds and dim same to input width.
    // NOTE: old matrix implementation of stride is different to eigen.
    // TODO(typhoonzero): launch kernel on specified stream, need sync?
    // TODO(typhoonzero): refine this kernel.
    dim3 threads(256, 1);
    dim3 grid(input_height, 1);

    KeMatrixTopK<T, 5, 256><<<grid, threads>>>(
        output_data, output->dims()[1], indices_data, input_data, input_width,
        input_width, int(k));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_GPU_KERNEL(top_k, paddle::operators::TopkOpCUDAKernel<float, int>);
