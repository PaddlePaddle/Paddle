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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"

namespace phi {

constexpr int32_t WARP_SIZE = 32; 
constexpr int32_t HALF_WARP = 16; 

template <typename D>
class PDDataTypeTraits{
 public:
  typedef D DataType;
};

template <>
class PDDataTypeTraits<phi::dtype::float16> {
 public:
  typedef half DataType;
};

template<typename T>
struct MaxFunc{
  __device__ T operator()(T a, T b){
    return max(a, b); 
  }
}; 

template<>
struct MaxFunc<half>{
  __device__ half operator()(half a, half b){
    return __hmax(a, b); 
  }
}; 

template<>
struct MaxFunc<half2>{
  __device__ half2 operator()(half2 a, half2 b){
    return __hmax2(a, b); 
  }
};


template<typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) Pack {
  __device__ Pack() {
    #pragma unroll
    for(int i = 0; i < VecSize; i++){
      elem[i] = 0.0f; 
    }
  }
  union {
    T elem[VecSize];
  };

  __device__ void pack_abs_max(Pack<T, VecSize> packA){
    #pragma unroll 
    for(int i = 0; i < VecSize; i++){
      elem[i] = max(elem[i], abs(packA.elem[i])); 
    }
  }
};

template <int VecSize>
struct alignas(sizeof(half) * VecSize) Pack<half, VecSize> {
  __device__ Pack() {
    #pragma unroll
    for(int i = 0; i < VecSize; i++){
      elem[i] = 0.0f; 
    }
  }
  union {
    half elem[VecSize];
    half2 elem[VecSize / 2]; 
  };

  __device__ void pack_abs_max(Pack<half, VecSize> packA){
    #pragma unroll 
    for(int i = 0; i < VecSize; i++){
      elem[i] = MaxFunc<half>()(elem[i], __habs(packA.elem[i])); 
    }
  }
};

template<typename T, int VecSize>
__device__ T PackReduceAbsMax(Pack<T, VecSize> pack){
    T res = 0.0; 
    #pragma unroll
    for(int i = 0; i < VecSize; i++){
      res = MaxFunc<T>()(res, pack.elem[i]); 
    }
    return res; 
}

template <typename T>
__inline__ __device__ T WarpReduceAbsMax(T val, unsigned lane_mask) {
  #pragma unroll
  for (int mask = HALF_WARP; mask > 0; mask >>= 1){
    val = MaxFunc<T>()(val, __shfl_xor_sync(lane_mask, val, mask, WARP_SIZE));
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceAbsMax(T val, unsigned mask) {
  static __shared__ T smem[WARP_SIZE]; 
  int32_t lane_id = threadIdx.x & 0x1f; 
  int32_t warp_id = threadIdx.x >> 5; 
  val = WarpReduceAbsMax(val, mask); 
  if(lane_id == 0){
    smem[warp_id] = val; 
  }
  __syncthreads(); 
  T abs_max_val = (threadIdx.x < blockDim.x / WARP_SIZE) ? smem[threadIdx.x] : static_cast<T>(0.0f); 
  abs_max_val = WarpReduceAbsMax(abs_max_val, mask); 
  return abs_max_val; 
}

template<typename T, int VecSize>
__global__ void ReduceAbsMaxKernel(const T* x, T* out, int32_t rows, int32_t cols){
  Pack<T, VecSize> abs_max_pack{}; 
  Pack<T, VecSize> load_pack{};
  using LoadType = Pack<T, VecSize>;
  T local_max_val = 0.0; 
  for(int row_idx = blockIdx.x; row_idx < rows; row_idx += gridDim.x){
      for(int col_idx = threadIdx.x * VecSize; col_idx < cols; col_idx += blockDim.x * VecSize){
          int32_t linear_index = row_idx * cols + col_idx; 
          const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
          load_pack = *x_load; 
          // printf("linearidx is: %d, Load pack val is: %f, %f, %f, %f \n", linear_index, load_pack.elem[0], load_pack.elem[1], load_pack.elem[2], load_pack.elem[3]); 
          abs_max_pack.pack_abs_max(load_pack); 
          // printf("linearidx is: %d, absmax pack val is: %f, %f, %f, %f \n", linear_index, abs_max_pack.elem[0], abs_max_pack.elem[1], abs_max_pack.elem[2], abs_max_pack.elem[3]); 
      }
      local_max_val = PackReduceAbsMax<T, VecSize>(abs_max_pack); 
      // printf("local max val is: %f \n", local_max_val); 
      T row_max_val = BlockReduceAbsMax<T>(local_max_val, 0xffffffff); 
      // printf("row max val is: %f \n", local_max_val); 
      if(threadIdx.x == 0){
        // printf("row max val is: %f \n", local_max_val); 
        out[row_idx] = row_max_val; 
      }
  }
}

// constexpr int32_t BlockSize = 128; 
// constexpr int32_t BlockSize = 256; 
constexpr int32_t BlockSize = 512; 

template <typename T, int VecSize>
bool TryLaunchKernel(const phi::GPUContext& dev_ctx,
                      const T* x_data, 
                      T* out_data, 
                      const int64_t rows, 
                      const int64_t cols){
  if((VecSize <= (16 / sizeof(T))) &&
     (cols % VecSize == 0) &&
     (reinterpret_cast<uintptr_t>(x_data) % sizeof(T) == 0) &&
     (reinterpret_cast<uintptr_t>(out_data) % sizeof(T) == 0)){
    ReduceAbsMaxKernel<T, VecSize><<<rows, BlockSize, 0, dev_ctx.stream()>>>(x_data, out_data, rows, cols);
    printf("Here is vecsize %d \n", VecSize);  
    return true; 
  }
  return false; 
}

template <typename T>
void DispatchVecSize(const phi::GPUContext& ctx,
                     const T* x_data, 
                     T* out_data, 
                     const int64_t rows, 
                     const int64_t cols) {
  using Func = bool (*)(const phi::GPUContext& ctx,
                        const T* x_data, 
                        T* out_data, 
                        const int64_t rows, 
                        const int64_t cols);
  Func funcs[] = {
      TryLaunchKernel<T, 8>,  
      TryLaunchKernel<T, 4>,  
      TryLaunchKernel<T, 2>,  
      TryLaunchKernel<T, 1>,  
  };

  for (int i = 0; i < 4; i++) {
    if (funcs[i](ctx, x_data, out_data, rows, cols)) {
      return;
    }
  }
}

template <typename T, typename Context>
void RowReduceAbsMaxKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  typedef PDDataTypeTraits<T> traits;
  typedef typename traits::DataType DataType;
  auto x_mat_dims = phi::flatten_to_2d(x.dims(), x.dims().size() - 1);
  const int64_t rows = x_mat_dims[0];
  const int64_t cols = x_mat_dims[1];
  const T* x_data = x.data<T>(); 
  T* out_data = out->data<T>();  
  DispatchVecSize<DataType>(dev_ctx, 
                            reinterpret_cast<const DataType*>(x_data), 
                            reinterpret_cast<DataType*>(out_data), rows, cols); 

//   template <typename T>
// void DispatchVecSize(const phi::GPUContext& ctx,
//                      const T* x_data, 
//                      T* out_data, 
//                      const int64_t rows, 
//                      const int64_t cols) 
}

}  // namespace phi

PD_REGISTER_KERNEL(row_reduce_absmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::RowReduceAbsMaxKernel,
                   phi::dtype::float16,
                   float) {}