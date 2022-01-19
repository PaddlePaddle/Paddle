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

#include <cooperative_groups.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/timer.h"

#include "paddle/fluid/operators/filter_by_instag_op.h"

namespace cg = cooperative_groups;

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
using Vector = framework::Vector<T>;
#else
template <typename T>
using Vector = framework::CPUVector<T>;
#endif

#define WARP_SIZE 32
#define MAX_WARP_NUM 32

template <typename T>
__global__ void filter_copy_fuse_kernel(
    const size_t N, const int ins_per_thread, size_t* x1_lods_data,
    size_t* x2_lods_data, const int64_t* x2_data, const int64_t* x3_data,
    int64_t filter_tag_size, T* out_data, int64_t* map_data,
    size_t* map_lods_data, size_t* out_lods_data, size_t* out_idx_data,
    const T* x1_data, int x1_embed_size, int x1_lods_filled, int x2_lods_filled,
    float* loss_weight_data, int fill_value) {
  // N is instance num
  // one threads for ins_per_thread(4) instances
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int ins_start = idx * ins_per_thread;
  int ins_end = (idx + 1) * ins_per_thread;

  if (ins_start >= N) return;
  if (N < ins_end) ins_end = N;

  if (!x1_lods_filled) {
    for (int p = ins_start; p < ins_end; p++) {
      x1_lods_data[p] = p;
    }
    if (idx == 0) {
      x1_lods_data[N] = N;
    }
  }

  if (!x2_lods_filled) {
    for (int p = ins_start; p < ins_end; p++) {
      x2_lods_data[p] = p;
    }
    if (idx == 0) {
      x2_lods_data[N] = N;
    }
  }

  __syncthreads();

  // extern __shared__ int shared_data[];
  // int* flag_data = shared_data;
  // int* prefix_sum_data = (int*)(&(flag_data[N]));
  // int* mmap_aux_data = (int*)(&(prefix_sum_data[N]));

  // ================== to be optimized =============================

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int flag_data[2];
  int prefix_sum_data[2];
  int gid = idx / WARP_SIZE;

  __shared__ int shr[MAX_WARP_NUM];
  __shared__ int shr2[MAX_WARP_NUM];
  __shared__ int shr3[MAX_WARP_NUM];

  for (int p = ins_start; p < ins_end; p++) {
    int ins_tag_start = x2_lods_data[p];
    int ins_tag_end = x2_lods_data[p + 1];

    flag_data[p - ins_start] = 0;

    // filter logic
    int i = ins_tag_start;
    for (; i < ins_tag_end; i++) {
      int64_t ins_tag = x2_data[i];
      int j = 0;
      for (; j < filter_tag_size; j++) {
        if (x3_data[j] == ins_tag) break;
      }
      // if ins_tag in filter tag
      if (j < filter_tag_size) {
        flag_data[p - ins_start] = 1;
        break;
      }
    }
  }

  for (int p = ins_start; p < ins_end; p++) {
    if (p == ins_start) {
      prefix_sum_data[p - ins_start] = 0;
    } else {
      prefix_sum_data[p - ins_start] =
          flag_data[p - ins_start] * (x1_lods_data[p] - x1_lods_data[p - 1]);
    }
  }

  // prefix
  int local_addr = prefix_sum_data[ins_end - 1 - ins_start];
  int sum_addr = local_addr;

  // flag
  int local_flag = 0;
  for (int p = ins_start; p < ins_end; p++) {
    local_flag += flag_data[p - ins_start];
  }
  int sum_flag = local_flag;

  // out_lods
  int local_out_lods = 0;
  for (int p = ins_start; p < ins_end; p++) {
    local_out_lods +=
        flag_data[p - ins_start] * (x1_lods_data[p + 1] - x1_lods_data[p]);
  }
  int sum_out_lods = local_out_lods;

  // warp reduce
  for (int i = 1; i < 32; i++) {
    sum_addr += g.shfl_up(sum_addr, i);
    sum_flag += g.shfl_up(sum_flag, i);
    sum_out_lods += g.shfl_up(sum_out_lods, i);
  }

  if (g.thread_rank() == 31) {
    shr[gid] = sum_addr;
    shr2[gid] = sum_flag;
    shr3[gid] = sum_out_lods;
  }

  b.sync();

  // communicate between warp
  //
  int sum_addr2 = shr[g.thread_rank()];
  int sum_flag2 = shr2[g.thread_rank()];
  int sum_out_lods2 = shr3[g.thread_rank()];

  for (int i = 1; i < 32; i++) {
    sum_addr2 += g.shfl_up(sum_addr2, i);
    sum_flag2 += g.shfl_up(sum_flag2, i);
    sum_out_lods2 += g.shfl_up(sum_out_lods2, i);
  }

  int sum_addr3 = g.shfl(sum_addr2, gid);
  int sum_flag3 = g.shfl(sum_flag2, gid);
  int sum_out_lods3 = g.shfl(sum_out_lods2, gid);

  int p_addr = sum_addr3 - shr[gid] + sum_addr - local_addr;
  int p_flag = sum_flag3 - shr2[gid] + sum_flag - local_flag;
  int p_out_lods = sum_out_lods3 - shr3[gid] + sum_out_lods - local_out_lods;

  for (int p = ins_start; p < ins_end; p++) {
    prefix_sum_data[p - ins_start] += p_addr;
  }

  // __syncthreads() only sync threads within block
  // so we let one thread process multi idx
  // __syncthreads();
  // thread with thread id = 0 compute prefix_sum array
  // prefix_sum array is shared memory
  // extern __shared__ int prefix_sum_data[];

  // if (idx == 0) {
  //  for (int i = 0; i < N; i++) {
  //    if (i == 0) {
  //      prefix_sum_data[i] = 0;
  //    } else {
  //      prefix_sum_data[i] =
  //          prefix_sum_data[i - 1] +
  //          flag_data[i - 1] * (x1_lods_data[i] - x1_lods_data[i - 1]);
  //    }
  //  }
  // }
  // __syncthreads();

  int out_lods_idx = p_flag + 1;

  for (int p = ins_start; p < ins_end; p++) {
    if (flag_data[p - ins_start] == 1) {
      size_t batch_len = x1_lods_data[p + 1] - x1_lods_data[p];
      int t = out_lods_idx - 1;
      out_lods_data[t] = p_out_lods;
      map_data[t * 3] = (int64_t)out_lods_data[t];
      map_data[t * 3 + 1] = x1_lods_data[p];
      map_lods_data[t] = t;

      out_lods_data[out_lods_idx] =
          out_lods_data[t] + (x1_lods_data[p + 1] - x1_lods_data[p]);
      map_data[t * 3 + 2] = out_lods_data[t + 1] - out_lods_data[t];

      out_lods_idx++;
    }
  }

  /*

    if (idx == 0) {
      int out_lods_idx = 1;
      for (int i = 0; i < N; i++) {
        if (flag_data[i] == 1) {
          size_t batch_len = x1_lods_data[i + 1] - x1_lods_data[i];
          mmap_aux_data[out_lods_data[out_lods_idx - 1]] = x1_lods_data[i];
          int p = out_lods_idx - 1;
          map_data[p * 3] = (int64_t)out_lods_data[p];
          map_data[p * 3 + 1] = mmap_aux_data[out_lods_data[p]];
          map_lods_data[p] = p;
          // map_data[p * 3 + 2] = out_lods_data[p + 1] - out_lods_data[p];
          out_lods_data[out_lods_idx] =
              out_lods_data[p] + (x1_lods_data[i + 1] - x1_lods_data[i]);
          map_data[p * 3 + 2] = out_lods_data[p + 1] - out_lods_data[p];
          out_lods_idx++;
        }
      }
      *out_idx_data = out_lods_idx;
      map_lods_data[out_lods_idx - 1] = out_lods_idx - 1;
    }

  */

  // =========== to be optimized =====================================

  // fill loss_weight_data
  if (*out_idx_data > 1) {
    int out_data_num = *out_idx_data - 1;
    int out_start = ins_start;
    if (out_start < out_data_num) {
      int out_end = ins_end > out_data_num ? out_data_num : ins_end;
      for (int p = out_start; p < out_end; p++) {
        loss_weight_data[p] = fill_value;
      }
    }
  }

  // 0 1 0 1 0 1 0 1
  // 0 1 3 6 10 15 21 28 36
  // 0 0 2 2 6 6
  //__syncthreads();

  for (int p = ins_start; p < ins_end; p++) {
    // copy logic
    if (flag_data[p - ins_start] == 1) {
      auto output_start_idx = prefix_sum_data[p - ins_start];

      T* dst = out_data + output_start_idx * x1_embed_size;

      const T* src_start = x1_data + x1_lods_data[p] * x1_embed_size;
      const T* src_end = x1_data + (x1_lods_data[p + 1]) * x1_embed_size;

      // optimized
      for (const T *j = src_start; j != src_end; dst++, j++) {
        *dst = *j;
      }
    }
  }
}

/*
template <>
void filter_copy_fuse_kernel<float>(
    const size_t N, const int ins_per_thread, size_t* x1_lods_data,
    size_t* x2_lods_data, const int64_t* x2_data, const int64_t* x3_data,
    int64_t filter_tag_size, float* out_data, int64_t* map_data, size_t*
map_lods_data, size_t* out_lods_data, size_t* out_idx_data, const float*
x1_data,
    int x1_embed_size, int x1_lods_filled, int x2_lods_filled) {

  // N is instance num
  // one threads for ins_per_thread(4) instances
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int ins_start = idx * ins_per_thread;
  int ins_end = (idx + 1) * ins_per_thread;

  if (ins_start >= N) return;
  if (N < ins_end) ins_end = N;

  if (!x1_lods_filled) {
    #pragma unroll
    for (int p = ins_start; p < ins_end; p++) {
      x1_lods_data[p] = p;
    }

    if (idx == 0) {
      x1_lods_data[N] = N;
    }
  }

  if (!x2_lods_filled) {
    #pragma unroll
    for (int p = ins_start; p < ins_end; p++) {
      x2_lods_data[p] = p;
    }
    if (idx == 0) {
      x2_lods_data[N] = N;
    }
  }

  __syncthreads();

  extern __shared__ int shared_data[];
  int* flag_data = shared_data;
  int* prefix_sum_data = (int*)(&(flag_data[N]));
  int* mmap_aux_data = (int*)(&(prefix_sum_data[N]));

  #pragma unroll
  for (int p = ins_start; p < ins_end; p++) {

    int ins_tag_start = x2_lods_data[p];
    int ins_tag_end = x2_lods_data[p + 1];

    flag_data[p] = 0;
    // filter logic
    int i = ins_tag_start;
    for (; i < ins_tag_end; i++) {

      int64_t ins_tag = x2_data[i];

      int j = 0;
      for (; j < filter_tag_size; j++) {
        if (x3_data[j] == ins_tag) break;
      }

      // if ins_tag in filter tag
      if (j < filter_tag_size) {
        flag_data[p] = 1;
        break;
      }

    }

  }

  // __syncthreads() only sync threads within block
  // so we let one thread process multi idx

  __syncthreads();

  // thread with thread id = 0 compute prefix_sum array
  // prefix_sum array is shared memory
  //extern __shared__ int prefix_sum_data[];

  if (idx == 0) {

    for (int i = 0; i < N; i++) {

      if (i == 0) {
        prefix_sum_data[i] = 0;
      } else {
        prefix_sum_data[i] =
            prefix_sum_data[i - 1] +
            flag_data[i - 1] * (x1_lods_data[i] - x1_lods_data[i - 1]);
      }

    }

  }

  __syncthreads();

  if (idx == 0) {

    int out_lods_idx = 1;

    for (int i = 0; i < N; i++) {

      if (flag_data[i] == 1) {

        auto batch_len = x1_lods_data[i + 1] - x1_lods_data[i];

        mmap_aux_data[out_lods_data[out_lods_idx - 1]] = x1_lods_data[i];

        int p = out_lods_idx - 1;
        map_data[p * 3] = (int64_t)out_lods_data[p];
        map_data[p * 3 + 1] = mmap_aux_data[out_lods_data[p]];

        map_lods_data[p] = p;
        //map_data[p * 3 + 2] = out_lods_data[p + 1] - out_lods_data[p];
        out_lods_data[out_lods_idx] = out_lods_data[p] + (x1_lods_data[i + 1] -
x1_lods_data[i]);

        map_data[p * 3 + 2] = out_lods_data[p + 1] - out_lods_data[p];
        out_lods_idx++;

      }

    }

    *out_idx_data = out_lods_idx;
    map_lods_data[out_lods_idx - 1] = out_lods_idx - 1;

  }

  // 0 1 0 1 0 1 0 1
  // 0 1 3 6 10 15 21 28 36
  // 0 0 2 2 6 6
  //__syncthreads();

  #pragma unroll
  for (int p = ins_start; p < ins_end; p++) {

    // copy logic
    if (flag_data[p] == 1) {

      auto output_start_idx = prefix_sum_data[p];

      float* dst = out_data + output_start_idx * x1_embed_size;

      const float* src_start = x1_data + x1_lods_data[p] * x1_embed_size;
      const float* src_end = x1_data + (x1_lods_data[p + 1]) * x1_embed_size;

      int data_num = (x1_lods_data[p + 1] - x1_lods_data[p]) * x1_embed_size;

      float4* dst_iter = reinterpret_cast<float4*>(dst);
      const float4* src_iter = reinterpret_cast<const float4*>(src_start);
      for(int t = 0; t < data_num / 4; t++) {
        *dst_iter = *src_iter;
        dst_iter++;
        src_iter++;
      }
      // optimized
      const float* j = reinterpret_cast<const float*>(src_iter);
      float* dst_final = reinterpret_cast<float*>(dst_iter);
      for (; j != src_end; dst_final++, j++) {
        *dst_final = *j;
      }

    }

  }


}


__global__ void fill_kernel(float* data, const int data_num, const int
ins_per_thread, float fill_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ins_start = idx * ins_per_thread;
    int ins_end = (idx + 1) * ins_per_thread;
    if (ins_start >= data_num) return;
    if (ins_end > data_num) ins_end = data_num;
    for (int p = ins_start; p < ins_end; p++) {
      data[p] = fill_value;
    }
}

*/

template <typename T>
__global__ void copy_grad_kernel(const size_t N, const int ins_per_thread,
                                 const T* out_grad_data, T* x1_grad_data,
                                 const int64_t* map_data, int x1_embed_size) {
  // N is instance num
  // one threads for one instance
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int ins_start = idx * ins_per_thread;
  int ins_end = (idx + 1) * ins_per_thread;
  if (ins_start >= N) {
    return;
  }
  if (ins_end > N) ins_end = N;
  for (int p = ins_start; p < ins_end; p++) {
    T* dst = x1_grad_data + map_data[p * 3 + 1] * x1_embed_size;
    const T* src_start = out_grad_data + map_data[p * 3] * x1_embed_size;
    const T* src_end =
        out_grad_data + (map_data[p * 3] + map_data[p * 3 + 2]) * x1_embed_size;
    for (const T *j = src_start; j != src_end; dst++, j++) {
      *dst = *j;
    }
  }
}

/*
template <>
void copy_grad_kernel<float>(const size_t N, const int ins_per_thread, const
float* out_grad_data,
                                 float* x1_grad_data, const int64_t* map_data,
                                 int x1_embed_size) {
  // N is instance num
  // one threads for one instance
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int ins_start = idx * ins_per_thread;
  int ins_end = (idx + 1) * ins_per_thread;

  if (ins_start >= N) {
    return;
  }
  if (ins_end > N) ins_end = N;

  #pragma unroll
  for (int p = ins_start; p < ins_end; p++) {

    float* dst = x1_grad_data + map_data[p * 3 + 1] * x1_embed_size;
    const float* src_start = out_grad_data + map_data[p * 3] * x1_embed_size;
    const float* src_end =
      out_grad_data +
      (map_data[p * 3] + map_data[p * 3 + 2]) * x1_embed_size;

    int data_num = map_data[p * 3 + 2] * x1_embed_size;

    const float4* src_iter = reinterpret_cast<const float4*>(src_start);
    float4* dst_iter = reinterpret_cast<float4*>(dst);

    for( int t = 0; t < data_num / 4; t++) {
      *dst_iter = *src_iter;
      dst_iter++;
      src_iter++;
    }
    const float* j = reinterpret_cast<const float*>(src_iter);
    float* dst_final = reinterpret_cast<float*>(dst_iter);

    // using float*
    for (; j != src_end; dst_final++, j++) {
      *dst_final = *j;
    }
  }
}
*/

template <typename T>
class FilterByInstagGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // platform::Timer timeline_;
    // timeline_.Start();

    const auto gpu_place =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace());
    gpuStream_t current_stream = context.cuda_device_context().stream();
    int max_thread_num_per_block =
        context.cuda_device_context().GetMaxThreadsPerBlock();
    // X1 is global FC output
    // Dim [batch size, embedding size]
    const LoDTensor* x1 = context.Input<LoDTensor>("Ins");
    bool is_lod = context.Attr<bool>("is_lod");

    int is_x1_lod = -1;
    if (is_lod)
      is_x1_lod = 1;
    else
      is_x1_lod = 0;

    int64_t out_val_if_empty = context.Attr<int64_t>("out_val_if_empty");
    size_t x1_embed_size = x1->dims()[1];
    // X2 is ins tag list
    // LoD [[0, Sum(ins1), Sum(ins1, ins2), ... ]]
    const LoDTensor* x2 = context.Input<LoDTensor>("Ins_tag");
    // expected auto = const int64_t
    const int64_t* x2_data = x2->data<int64_t>();

    // X3 is local fc tag list
    // LoD [[0, Sum(fc1), Sum(fc1, fc2) ...]]
    const Tensor* x3 = context.Input<Tensor>("Filter_tag");
    const int64_t* x3_data = x3->data<int64_t>();

    // =========== need to be further optimized =================

    int x2_lods_filled = 0;

    Vector<size_t> x2_lods;
    // Vector, in GPU
    if (x2->lod().size() != 0) {  // lod_level = 1
      x2_lods = x2->lod()[0];
      x2_lods_filled = 1;
    } else {  // lod_level = 0
      // const size_t x2_lods_size = x2->dims()[0];
      x2_lods.resize(x2->dims()[0] + 1);
      // move to cuda
      // x2_lods.push_back(0);
      // for (size_t i = 0; i < x2_lods_size; i++) {
      //  x2_lods.push_back(i + 1);
      //}
    }

    size_t* x2_lods_data = x2_lods.CUDAMutableData(gpu_place);
    const size_t x2_lods_size = x2_lods.size() - 1;
    // Vector, in GPU
    int x1_lods_filled = 0;
    Vector<size_t> x1_lods;
    if (!is_x1_lod) {
      // move to cuda
      x1_lods.resize(x1->dims()[0] + 1);
      // x1_lods.push_back(0);
      // for (int i = 0; i < x1->dims()[0]; i++) {
      //  x1_lods.push_back(i + 1);
      //}
    } else {
      // x1_lods = context.Input<LoDTensor>("Ins")->lod()[0];
      // new: lod_level=0 => lod() return {}
      if (x1->lod().size() != 0) {  // lod_level = 1
        x1_lods_filled = 1;
        x1_lods = x1->lod()[0];

      } else {  // lod_level = 0
        x1_lods.resize(x1->dims()[0] + 1);

        // move to cuda
        // x1_lods.push_back(0);
        // for (int i = 0; i < x1->dims()[0]; i++) {
        //  x1_lods.push_back(i + 1);
        //}
      }
    }

    size_t* x1_lods_data = x1_lods.CUDAMutableData(gpu_place);
    auto* x1_data = x1->data<T>();

    // ============= need to be further optimized ===================

    // timeline_.Pause();
    // std::cout << "prephase 1 cost time: " << timeline_.ElapsedSec() <<
    // std::endl;

    // timeline_.Start();
    // set output value
    // for those whose ins been dropout, set 0 for whole lines.
    // otherwise, copy whole line
    // Dim [local fc count, batch size, embedding size]
    LoDTensor* out = context.Output<LoDTensor>("Out");
    LoDTensor* map = context.Output<LoDTensor>("IndexMap");
    LoDTensor* loss_weight = context.Output<LoDTensor>("LossWeight");

    int out_first = x1->dims()[0];
    if (x1_lods_filled) {
      out_first = x1_lods.back();
    }

    // if (x1_lods_filled) {
    //  out->Resize(framework::make_ddim(
    //    {(int64_t)x1_lods.back(), (int64_t)x1_embed_size}));
    //} else {
    out->Resize(
        framework::make_ddim({(int64_t)out_first, (int64_t)x1_embed_size}));

    //}
    map->Resize(framework::make_ddim({(int64_t)x2_lods_size, 3}));

    loss_weight->Resize(framework::make_ddim({(int64_t)x2_lods_size, 1}));

    // timeline_.Pause();
    // std::cout << "prephase 2 cost time: " << timeline_.ElapsedSec() <<
    // std::endl;

    // timeline_.Start();
    // loss_weight->Resize(
    //      framework::make_ddim({(int64_t)x2_lods_size, 1}));

    T* out_data = out->mutable_data<T>(gpu_place);
    int64_t* map_data = map->mutable_data<int64_t>(gpu_place);

    float* loss_weight_data = loss_weight->mutable_data<float>(gpu_place);

    // this is not needed if implementation is correct
    // thrust::device_ptr<T> out_data_ptr(out_data);
    // thrust::fill(out_data_ptr, out_data_ptr + out->numel(), 0);

    // timeline_.Pause();
    // std::cout << "prephase 3 cost time: " << timeline_.ElapsedSec() <<
    // std::endl;

    // timeline_.Start();
    // std::cout << "=====DEBUG====== out numel " << out->numel() << " " <<
    // x2_lods_size << " " << x1_embed_size <<std::endl;

    // not needed
    // Vector<int> flag(x2_lods_size, 0);
    // int* flag_data = flag.CUDAMutableData(context.GetPlace());

    // Vector<int> prefix_sum(x2_lods_size, 0);
    // int* prefix_sum_data = prefix_sum.CUDAMutableData(context.GetPlace());

    int block_size = max_thread_num_per_block;
    int ins_per_thread = (x2_lods_size + block_size - 1) / block_size;
    //    x2_lods_size / block_size >= 1 ? x2_lods_size / block_size + 1: 1;
    dim3 block_dim(block_size);
    dim3 grid_dim(1);

    // std::cout << "=====DEBUG====== out numel " << out->numel() << " " <<
    // x2_lods_size << " " << x1_embed_size << " " << block_size << " " <<
    // ins_per_thread <<std::endl;

    // fileter_logic
    // filter_by_instag_cuda_kernel<<<grid_dim, block_dim, 0, current_stream>>>(
    //    x2_lods_size, x2_lods_data, x2_data, x3_data, x3->numel(), flag_data);

    // filter + copy fuse
    // std::unordered_map<int64_t, int64_t> mmap_aux;

    // thrust::device_vector<int64_t> mmap_aux(x1_lods.back());

    // shared_data
    // Vector<int64_t> mmap_aux(x1_lods.back()); // space -> time

    // ============reduce time copy data from cpu to gpu================
    Vector<size_t> out_lods(x2_lods_size + 1, 0);
    Vector<size_t> map_lods(x2_lods_size + 1, 0);

    // thrust::device_vector<size_t> out_idx(1);
    Vector<size_t> out_idx(1, 0);

    // out_lods.resize(x2_lods_size + 1);
    // out_lods[0] = 0;

    size_t* out_lods_data = out_lods.CUDAMutableData(gpu_place);
    size_t* map_lods_data = map_lods.CUDAMutableData(gpu_place);

    // size_t* out_idx_data = thrust::raw_pointer_cast(&out_idx[0]);
    // int64_t* mmap_aux_data = thrust::raw_pointer_cast(&mmap_aux[0]);

    // int64_t* mmap_aux_data = mmap_aux.CUDAMutableData(context.GetPlace());
    size_t* out_idx_data = out_idx.CUDAMutableData(gpu_place);
    // ==================================================================

    // auto* out_lods_data = out_lods->mutable_data<size_t>(context.GetPlace());
    // auto* mmap_aux_data =
    // mmap_aux->mutable_data<int64_t>(context.GetPlace());
    // auto* out_idx_data = out_idx->mutable_data<size_t>(context.GetPlace());

    // timeline_.Pause();
    // std::cout << "pre phase 4 cost time: " << timeline_.ElapsedSec() <<
    // std::endl;

    // timeline_.Start();

    // filter_copy_fuse_kernel<<<grid_dim, block_dim, 0,
    //                          current_stream>>>(
    //    x2_lods_size, ins_per_thread, x1_lods_data, x2_lods_data, x2_data,
    //    x3_data, x3->numel(), flag_data, out_data, map_data, map_lods_data,
    //    out_lods_data, mmap_aux_data, out_idx_data, x1_data, x1_embed_size);

    float fill_value = 1.0;

    filter_copy_fuse_kernel<<<grid_dim, block_dim,
                              out_first * sizeof(int) +
                                  x2_lods_size*(sizeof(int) + sizeof(int)),
                              current_stream>>>(
        x2_lods_size, ins_per_thread, x1_lods_data, x2_lods_data, x2_data,
        x3_data, x3->numel(), out_data, map_data, map_lods_data, out_lods_data,
        out_idx_data, x1_data, x1_embed_size, x1_lods_filled, x2_lods_filled,
        loss_weight_data, fill_value);

    // std::cout << "============DEBUG=============flag data" << std::endl;
    // for(int i = 0; i < x2_lods_size; i++) {
    //  std::cout << flag_data[i] << " ";
    //}
    // std::cout << std::endl;

    // filter + copy fuse
    // copy_kernel<<<grid_dim_2, block_dim_2, 0, current_stream>>>(N, out_data,
    // x1_data, map_data, x1_embed_size);

    platform::GpuStreamSync(current_stream);
    // timeline_.Pause();
    // std::cout << "kernel phase cost time: " << timeline_.ElapsedSec() <<
    // std::endl;

    // timeline_.Start();
    // out_lods resize
    out_lods.resize(out_idx[0]);

    // timeline_.Pause();
    // std::cout << "kernel phase cost time: " << timeline_.ElapsedSec() << " "
    // << out_lods.size() << " " << out_idx[0] << std::endl;

    // timeline_.Start();
    // std::cout << "============DEBUG=============flag data" << std::endl;
    // thrust::device_ptr<int> flag_data_ptr(flag_data);

    // for(int i = 0; i < x2_lods_size; i++) {
    // std::cout << flag_data_ptr[i] << " ";
    //}
    // std::cout << std::endl;

    // thrust::device_ptr<const size_t> x1_lods_data_ptr(x1_lods_data);
    // std::cout << "============DEBUG=============x2_lods data" << std::endl;
    // for(int i = 0; i <= x2_lods_size; i++) {
    //  std::cout << x1_lods_data_ptr[i] << " ";
    //}
    // std::cout << std::endl;

    // std::unordered_map<int64_t, int64_t> mmap_aux;
    // Vector<size_t> out_lods;
    // out_lods.reserve(x2_lods_size + 1);
    // out_lods.push_back(0);

    // int cnt = 0;
    // for (auto it = flag.begin(); it != flag.end(); cnt++, it++) {
    //  if ((*it) == 1) {
    //    size_t batch_len = x1_lods[cnt + 1] - x1_lods[cnt];
    //    mmap_aux[out_lods.back()] = x1_lods[cnt];
    //    //out_lods.push_back(out_lods.back() + batch_len);
    //  }
    //}

    if (out_lods.size() - 1 > 0) {
      out->Resize(framework::make_ddim(
          {(int64_t)out_lods.back(), (int64_t)x1_embed_size}));

      map->Resize(framework::make_ddim({(int64_t)out_lods.size() - 1, 3}));
      loss_weight->Resize(
          framework::make_ddim({(int64_t)out_lods.size() - 1, 1}));

    } else {
      out->Resize(framework::make_ddim({1, (int64_t)x1_embed_size}));
      map->Resize(framework::make_ddim({1, 3}));
      loss_weight->Resize(framework::make_ddim({1, 1}));
    }

    // auto* out_data = out->mutable_data<T>(context.GetPlace());
    // auto* map_data = map->mutable_data<int64_t>(context.GetPlace());

    // float* loss_weight_data =
    //    loss_weight->mutable_data<float>(gpu_place);

    if (out_lods.size() - 1 > 0) {
      // move to cuda kernel
      // Vector<size_t> map_lods(out_lods.size(), 0);

      map_lods.resize(out_lods.size());
      // thrust::device_ptr<int64_t> map_data_ptr(map_data);
      // only one host -> device
      // thrust::host_vector<int64_t> h_vec(3 * (out_lods.size() - 1));

      // for (size_t i = 0; i < out_lods.size() - 1; i++) {

      // h_vec[i * 3] = (int64_t)out_lods[i];
      // h_vec[i * 3 + 1] = mmap_aux_data[(int64_t)out_lods[i]];
      // h_vec[i * 3 + 2] = out_lods[i + 1] - out_lods[i];
      //  map_lods[i] = i;

      //}

      // map_lods[out_lods.size() - 1] = out_lods.size() - 1;

      // only one copy
      // thrust::copy(h_vec.begin(), h_vec.end(), map_data_ptr);

      std::vector<Vector<size_t>> map_lod_info;
      map_lod_info.emplace_back(map_lods);
      map->set_lod(map_lod_info);
      loss_weight->set_lod(map_lod_info);
      std::vector<Vector<size_t>> out_lod_info;
      out_lod_info.emplace_back(out_lods);
      out->set_lod(out_lod_info);

      // std::cout << "=====DEBUG====== out numel " << out->numel() << " " <<
      // out_lods.back() << " " << x1_embed_size << std::endl;
      // thrust::device_ptr<T> out_data_ptr(out_data);
      // thrust::fill(out_data_ptr, out_data_ptr + out->numel(), 0);
      // can be optimized ???

      // move to cuda kernel

      // ins_per_thread = (loss_weight->numel() + block_size - 1) / block_size;
      // float fill_value = 1.0;
      // fill_kernel<<<grid_dim, block_dim, 0,
      // current_stream>>>(loss_weight_data, loss_weight->numel(),
      // ins_per_thread, fill_value);

      // platform::GpuStreamSync(current_stream);

      // thrust::device_ptr<float> loss_weight_data_ptr(loss_weight_data);
      // thrust::fill(loss_weight_data_ptr,
      //             loss_weight_data_ptr + loss_weight->numel(), 1.0);

      // only one kernel launch
      // size_t N = out_lods.size() - 1;
      // dim3 block_dim_2(block_size);
      // dim3 grid_dim_2((N + block_size - 1) / block_size);
      // copy_kernel<<<grid_dim_2, block_dim_2, 0, current_stream>>>(N,
      // out_data, x1_data, map_data, x1_embed_size);
      // cudaStreamSynchronize(current_stream);

      // timeline_.Pause();
      // std::cout << "left phase cost time: " << timeline_.ElapsedSec() <<
      // std::endl;

    } else {
      Vector<size_t> map_lods(2, 0);
      thrust::device_ptr<int64_t> map_data_ptr(map_data);
      map_data_ptr[0] = 0;
      map_data_ptr[1] = 1;
      map_data_ptr[2] = 1;
      map_lods[0] = 0;
      map_lods[1] = 1;
      out_lods.push_back(1);
      std::vector<Vector<size_t>> map_lod_info;
      map_lod_info.emplace_back(map_lods);
      map->set_lod(map_lod_info);
      loss_weight->set_lod(map_lod_info);
      std::vector<Vector<size_t>> out_lod_info;
      out_lod_info.emplace_back(out_lods);
      out->set_lod(out_lod_info);
      thrust::device_ptr<T> out_data_ptr(out_data);
      // gpu kernel
      if (std::is_same<T, int32_t>::value) {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<int32_t>(out_val_if_empty));
      } else if (std::is_same<T, int64_t>::value) {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<int64_t>(out_val_if_empty));
      } else if (std::is_same<T, float>::value) {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<float>(out_val_if_empty));
      } else {
        thrust::fill(out_data_ptr, out_data_ptr + out->numel(),
                     static_cast<double>(out_val_if_empty));
      }
      thrust::device_ptr<float> loss_weight_data_ptr(loss_weight_data);
      loss_weight_data_ptr[0] = 0;
    }
  }
};

template <typename T>
class FilterByInstagGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // platform::Timer timeline_;
    // timeline_.Start();

    const auto gpu_place =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace());
    gpuStream_t current_stream = context.cuda_device_context().stream();
    auto max_thread_num_per_block =
        context.cuda_device_context().GetMaxThreadsPerBlock();
    auto* output_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x1_grad = context.Output<LoDTensor>(framework::GradVarName("Ins"));
    auto* loss_weight = context.Input<LoDTensor>("LossWeight");
    auto* mmap = context.Input<LoDTensor>("IndexMap");
    auto* x1 = context.Input<LoDTensor>("Ins");

    x1_grad->set_lod(context.Input<LoDTensor>("Ins")->lod());
    x1_grad->Resize(x1->dims());

    auto* mmap_data = mmap->data<int64_t>();
    // expected auto = T
    auto* output_grad_data = output_grad->data<T>();
    auto* loss_weight_data = loss_weight->data<float>();

    // timeline_.Pause();

    // std::cout << "fill phase cost time: " << timeline_.ElapsedSec() <<
    // std::endl;

    // timeline_.Start();

    // expected auto = T
    auto* x1_grad_data = x1_grad->mutable_data<T>(gpu_place);
    thrust::device_ptr<T> x1_grad_data_ptr(x1_grad_data);
    thrust::device_ptr<const float> loss_weight_data_ptr(loss_weight_data);

    thrust::fill(x1_grad_data_ptr,
                 x1_grad_data_ptr + x1->dims()[0] * x1->dims()[1], 0);

    if (loss_weight->numel() != 1 || loss_weight_data_ptr[0] != 0) {
      auto output_dims = output_grad->dims();
      int x1_embed_size = output_dims[1];

      // one thread for multi-instances
      int block_size = max_thread_num_per_block;

      size_t N = mmap->dims()[0];
      dim3 block_dim(block_size);
      dim3 grid_dim((N + block_size - 1) / block_size);

      const int ins_per_thread =
          1;  // N / block_size >= 1 ? N / block_size + 1: 1;

      copy_grad_kernel<<<grid_dim, block_dim, 0, current_stream>>>(
          N, ins_per_thread, output_grad_data, x1_grad_data, mmap_data,
          x1_embed_size);

      cudaStreamSynchronize(current_stream);

      // timeline_.Pause();
      // std::cout << "grad kernel phase cost time: " << timeline_.ElapsedSec()
      // << std::endl;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(filter_by_instag, ops::FilterByInstagGPUKernel<float>,
                        ops::FilterByInstagGPUKernel<double>,
                        ops::FilterByInstagGPUKernel<int32_t>,
                        ops::FilterByInstagGPUKernel<int64_t>);

REGISTER_OP_CUDA_KERNEL(filter_by_instag_grad,
                        ops::FilterByInstagGradGPUKernel<float>,
                        ops::FilterByInstagGradGPUKernel<double>,
                        ops::FilterByInstagGradGPUKernel<int32_t>,
                        ops::FilterByInstagGradGPUKernel<int64_t>);
