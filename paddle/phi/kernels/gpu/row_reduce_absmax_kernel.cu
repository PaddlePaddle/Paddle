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


template<typename T>
struct AbsFunc{
  __device__ T operator()(T x){
    return abs(x); 
  }
}; 

template<>
struct AbsFunc<half>{
  __device__ half operator()(half x){
    return __habs(x); 
  }
}; 

template<>
struct AbsFunc<half2>{
  __device__ half2 operator()(half2 x){
    return __habs2(x); 
  }
};

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

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
    PackType<T, VecSize> storage;
  };

  __device__ void pack_abs_max(Pack<T, VecSize> packA){
    #pragma unroll 
    for(int i = 0; i < VecSize; i++){
      elem[i] = MaxFunc<T>()(elem[i], AbsFunc<T>()(packA.elem[i])); 
    }
  }
};

template <>
struct alignas(sizeof(half) * 1) Pack<half, 1> {
  __device__ Pack() {
    #pragma unroll
    for(int i = 0; i < 1; i++){
      elem[i] = 0.0f; 
    }
  }
  union {
    half elem[1];
    PackType<half, 1> storage;
  };

  __device__ void pack_abs_max(Pack<half, 1> packA){
    #pragma unroll 
    for(int i = 0; i < 1; i++){
      elem[i] = MaxFunc<half>()(elem[i], AbsFunc<half>()(packA.elem[i])); 
    }
  }
};

template <int VecSize>
struct alignas(sizeof(half2) * VecSize / 2) Pack<half, VecSize> {
  __device__ Pack() {
    #pragma unroll
    for(int i = 0; i < VecSize / 2; i++){
      elem_pack2[i] = make_half2(0.0, 0.0); 
    }
  }

  union {
    half2 elem_pack2[VecSize / 2];
    PackType<half, VecSize> storage;
  };

  __device__ void pack_abs_max(const Pack<half, VecSize>& packA){
    #pragma unroll 
    for(int i = 0; i < VecSize / 2; i++){
      elem_pack2[i] = MaxFunc<half2>()(elem_pack2[i], AbsFunc<half2>()(packA.elem_pack2[i])); 
    }
  }
};

template<typename T, int VecSize>
__device__ T PackReduceAbsMax(const Pack<T, VecSize>& pack){
    T res = 0.0; 
    #pragma unroll
    for(int i = 0; i < VecSize; i++){
      res = MaxFunc<T>()(res, pack.elem[i]); 
    }
    return res; 
}

template<>
__device__ half PackReduceAbsMax(const Pack<half, 1>& pack){
    half res = 0.0; 
    #pragma unroll
    for(int i = 0; i < 1; i++){
      res = MaxFunc<half>()(res, pack.elem[i]); 
    }
    return res; 
}

template<typename T, int VecSize>
__device__ typename std::enable_if<std::is_same<T, half>::value && VecSize % 2 == 0, half>::type 
PackReduceAbsMax(const Pack<half, VecSize>& pack){
    half2 res = make_half2(0.0, 0.0); 
    #pragma unroll
    for(int i = 0; i < VecSize / 2; i++){
      res = MaxFunc<half2>()(res, pack.elem_pack2[i]); 
    }
    return MaxFunc<half>()(res.x, res.y); 
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
  using LoadType = PackType<T, VecSize>;

  T local_max_val = 0.0; 
  for(int row_idx = blockIdx.x; row_idx < rows; row_idx += gridDim.x){
      for(int col_idx = threadIdx.x * VecSize; col_idx < cols; col_idx += blockDim.x * VecSize){
          int32_t linear_index = row_idx * cols + col_idx; 
          const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
          load_pack.storage = *x_load; 
          abs_max_pack.pack_abs_max(load_pack); 
      }
      local_max_val = PackReduceAbsMax<T, VecSize>(abs_max_pack); 
      T row_max_val = BlockReduceAbsMax<T>(local_max_val, 0xffffffff); 
      if(threadIdx.x == 0){
        out[row_idx] = row_max_val; 
      }
  }
}

constexpr int32_t BlockSize = 256; 

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
}

}  // namespace phi

PD_REGISTER_KERNEL(row_reduce_absmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::RowReduceAbsMaxKernel,
                   float, 
                   phi::dtype::float16) {}