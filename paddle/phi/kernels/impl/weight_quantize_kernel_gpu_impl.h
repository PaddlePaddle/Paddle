/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// #include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
namespace phi {

__global__ void weight_permute_kernel_wint8(
  const int8_t* input_data_dev,
  int8_t* output_data_dev,
  int numel,
  int total_k,
  int total_n){
    for(int linear_idx = blockIdx.x * blockDim.x + threadIdx.x; 
        linear_idx < numel; 
        linear_idx += blockDim.x * gridDim.x){
        int k_id = linear_idx / total_n;
        int n_id = linear_idx % total_n;
        constexpr int k_permute_const = 8;
        int k_mod_16 = k_id % 16;
        int temp_k_expr_1 = k_mod_16 - k_mod_16 / 8 * 8;
        int temp_k_expr_2 = k_mod_16 / 8;
        int permute_kk = temp_k_expr_1 
                            + temp_k_expr_2
                            + (temp_k_expr_2+1) % 2 * k_mod_16 * 2 / 2
                            + temp_k_expr_1*temp_k_expr_2 + k_id / 16 * 16;
        int permute_index = permute_kk % 64 
                              + permute_kk / 64*128 
                              + 64 * (n_id % 2) + total_k * 2 * (n_id / 2);
        int8_t shift_quant_weight = static_cast<int8_t>(
          static_cast<int32_t>(input_data_dev[linear_idx]) + 128
        );
        // printf("%d-%d, %d-%d, %d-%d\n",threadIdx.x, blockIdx.x, permute_index, linear_idx, k_id, n_id);
        output_data_dev[permute_index] = shift_quant_weight;
    }
}


template<typename GPUContext>
  void weight_permute_gpu(
    const GPUContext& dev_ctx,
    int8_t* input_data,
    int8_t* output_data,
    std::vector<int>& shape
    ){
    auto total_k = shape[0];
    auto total_n = shape[1];
    auto numel = total_k*total_n;
    auto gpu_config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, 1);
    int grid_size = gpu_config.GetGridSize();
    int block_size = gpu_config.GetBlockSize();
    VLOG(0)<<"### total_k: "<<total_k<<" total_n: "<<total_n;
    VLOG(0)<<"### grid_size: "<<grid_size<<" block_size: "<<block_size;
    weight_permute_kernel_wint8<<<grid_size,block_size>>>(
      input_data,
      output_data,
      numel,
      total_k,
      total_n
    );
  }

  template<typename T>
  inline __device__ T half_abs(T a){
    return a>static_cast<T>(0.0f)?a:-a;
  }
  

  template<typename T>
  inline __device__ T half_max(T a,T b){
    return a>b?a:b;
  }
  template<typename T, int VectorSize=8>
  __global__ void per_channel_quant_gpu(const T* weight_data,
                                          int8_t* quanted_weight_data,
                                          T* scale_data,
                                          int total_k,
                                          int total_vec_n){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n<total_vec_n){
      // printf("#### %d-%d-%d-%d \n", threadIdx.x, blockIdx.x,n, total_vec_n);
      const int4* vec_weight_data_ptr = reinterpret_cast<const int4*>(weight_data);
      int4* vec_scale_data = reinterpret_cast<int4*>(scale_data);
      int2* vec_quanted_weight_data = reinterpret_cast<int2*>(quanted_weight_data);
      phi::AlignedVector<T,VectorSize> abs_max;
      #pragma unroll
      for(int i = 0; i< VectorSize; ++i){
        abs_max[i] = static_cast<T>(0.0f);
      }
      #pragma unroll
      for(int k=0;k<total_k;++k){
        int linear_index = k*total_vec_n+n;
        phi::AlignedVector<T,VectorSize> weight;
        *(int4*)(&weight) = vec_weight_data_ptr[linear_index];
        // if(k == 0){
        //   printf("#### %d-%d-%d %f-%f \n", threadIdx.x, blockIdx.x, n, 
        //                                    static_cast<float>(weight[0]), static_cast<float>(weight[VectorSize-1]));
        // }
          #pragma unroll
          for(int i=0;i<VectorSize;++i){
              abs_max[i] = half_max((abs_max[i]),half_abs((weight[i])));
          }
        }
      phi::AlignedVector<T,VectorSize> scale;
      #pragma unroll
      for(int i=0;i<VectorSize;++i){
        scale[i] = abs_max[i]/static_cast<T>(127.0);
      }
      *(int4*)(&vec_scale_data[n]) = *(int4*)(&scale);
      for(int k=0;k<total_k;++k){
        phi::AlignedVector<int8_t,VectorSize> quanted_weight;
        int linear_index = k*total_vec_n+n;
        phi::AlignedVector<T,VectorSize> weight;
        *(int4*)(&weight) = *(int4*)(&vec_weight_data_ptr[linear_index]);
        #pragma unroll
        for(int i=0;i<VectorSize;++i){
          float scaled_weight = round(static_cast<float>(weight[i])/static_cast<float>(abs_max[i])*static_cast<float>(127.0));
          int8_t clipped_weight = static_cast<int8_t>(std::max(-127.f, std::min(127.f, scaled_weight)));
          quanted_weight[i] = clipped_weight;
        }
        // *(int2*)(&quanted_weight_data[linear_index]) = static_cast<int8_t>(weight_data[linear_index] / abs_max * 127.0);
        *(int2*)(&vec_quanted_weight_data[linear_index]) = *(int2*)(&quanted_weight);
      }
    }
  }

  template<typename T, typename GPUContext>
  void weight_quant_gpu(
                        const GPUContext& dev_ctx,
                        const T* weight_data,
                        int8_t* quanted_weight_data,
                        T* scale_data,
                        std::vector<int>& shape){
    int total_k = shape[0];
    int total_n = shape[1];
    VLOG(0)<<"shape : "<<shape[0]<<"-"<<shape[1];
    int numel = total_k*total_n;
    constexpr int kWarpSize = 32;
    constexpr int kBlockSize = 64;
    constexpr int kWarpNum = kBlockSize / kWarpSize;
    constexpr int kVectorSize = 128 / sizeof(T) / 8;
    int vec_total_n = total_n/kVectorSize;
    VLOG(0)<<"vec_total_n:"<<vec_total_n<<" kVectorSize:"<<kVectorSize;
    int kGridSize = max(vec_total_n/kBlockSize,(int)1);
    // DenseTensor abs_max;
    // abs_max.Resize({total_n});
    // T* abs_max_data = dev_ctx.template Alloc<T>(&abs_max);
    per_channel_quant_gpu<T, kVectorSize><<<kGridSize, kBlockSize>>>(
      weight_data,
      quanted_weight_data,
      scale_data,
      total_k,
      vec_total_n
    );

  }
}  // namespace phi
