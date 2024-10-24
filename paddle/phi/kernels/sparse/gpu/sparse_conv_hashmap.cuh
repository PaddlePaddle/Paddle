#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/transpose_function.cu.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"

/** Reserved value for indicating "empty". */
#define EMPTY_CELL (0)
/** CUDA naive thread block size. */
#define BLOCK_SIZE (256)

__inline__ __device__ int8_t atomicCAS(int8_t* address, int8_t compare, int8_t val) {
  int32_t* base_address = (int32_t*)((char*)address - ((size_t)address & 3));
  int32_t int_val = (int32_t)val << (((size_t)address & 3) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 3) * 8);
  return (int8_t)atomicCAS(base_address, int_comp, int_val);
}

// TODO: can we do this more efficiently?
__inline__ __device__ int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val) {
  int32_t* base_address = (int32_t*)((char*)address - ((size_t)address & 2));
  int32_t int_val = (int32_t)val << (((size_t)address & 2) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 2) * 8);
  return (int16_t)atomicCAS(base_address, int_comp, int_val);
}

__inline__ __device__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val) {
  return (int64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                            (unsigned long long)val);
}

template <typename dtype=int>
__device__ uint64_t hash_func_64b(dtype* data, int n=4){
  uint64_t hash = 14695981039346656037UL;
  for (int j = 0; j < n; j++) {
    hash ^= (unsigned int)data[j];
    hash *= 1099511628211UL;
  }
  // hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
  return hash;
}

template <typename key_type>
__device__ int hash(key_type key, int _capacity){
  return (uint64_t)key % _capacity;
}

template <typename key_type, typename val_type>
class GPUHashTable {
 private:
 //public:
  bool free_pointers;
  const int _capacity;
  const int _divisor;
  const int _width;
  key_type* table_keys;
  val_type* table_vals;
  void insert_many_coords(const phi::GPUContext& dev_ctx, const int *coords, const int n);
  void lookup_many_coords(const phi::GPUContext& dev_ctx, const int *coords, val_type *results,
    const int* kernel_sizes, const int* tensor_strides,
    const int n, const int kernel_volume);
 public:
  GPUHashTable(phi::DenseTensor* table_keys, phi::DenseTensor* table_vals, const int divisor, const int width)
      : _capacity(table_keys->dims()[0]), free_pointers(false), table_keys(table_keys->data<key_type>()),
      table_vals(table_vals->data<val_type>()), _divisor(divisor), _width(width){};
  ~GPUHashTable() {
    if(free_pointers){
      cudaFree(table_keys);
      cudaFree(table_vals);
    }
  };
  void insert_coords(const phi::GPUContext& dev_ctx, const phi::DenseTensor& coords);
  void lookup_coords(const phi::GPUContext& dev_ctx, const phi::DenseTensor& coords, const int* kernel_sizes, const int* tensor_strides, int kernel_volume, phi::DenseTensor* results);
  int get_divisor(){return _divisor;}
  int get_capacity(){return _capacity;}
};

using hashtable = GPUHashTable<int64_t, int>;
using hashtable32 = GPUHashTable<int, int>;

template <typename key_type=int64_t, typename val_type=int>
__global__ void insert_coords_kernel(key_type* table_keys, val_type* table_vals, const int* coords, int n, int _capacity, int _width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        key_type key = (key_type)(hash_func_64b(coords + idx*_width, _width));
        int value = idx + 1;
        int slot = hash(key, _capacity);
        while (true)
        {
            key_type prev = atomicCAS(&table_keys[slot], EMPTY_CELL, key);
            if (prev == EMPTY_CELL || prev == key)
            {
                table_vals[slot] = value;
                return;
            }
            slot = (slot + 1) % _capacity;
        }
    }
}


template <typename key_type=int64_t, typename val_type=int, bool odd>
__global__ void lookup_coords_kernel(
  key_type* table_keys, val_type* table_vals, const int* coords, val_type* vals,
  const int* kernel_sizes, const int* strides,
  int n, int _capacity, int kernel_volume, int _width)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tidx / kernel_volume;
    int _kernel_idx = tidx % kernel_volume;
    int kernel_idx = _kernel_idx;
    const int* in_coords = coords + _width * idx;
    int coords_out[4];
    //coords_out[2] = in_coords[2];
    //coords_out[3] = in_coords[3];
    coords_out[0] = in_coords[0];

    if constexpr (odd)
    {
      #pragma unroll
      for(int i = 0; i <= _width-2; i++){
        int cur_offset = _kernel_idx % kernel_sizes[i];
        cur_offset -= (kernel_sizes[i] - 1) / 2;
        coords_out[i+1] = in_coords[i+1] * strides[i] + cur_offset;
        _kernel_idx /= kernel_sizes[i];
      }
    }
    else
    {
      #pragma unroll
      for(int i = _width-2; i >= 0; i--){
        int cur_offset = _kernel_idx % kernel_sizes[i];
        cur_offset -= (kernel_sizes[i] - 1) / 2;
        coords_out[i+1] = in_coords[i+1] * strides[i] + cur_offset;
        _kernel_idx /= kernel_sizes[i];
      }
    }

    if (idx < n)
    {
        key_type key = (key_type)(hash_func_64b(coords_out, _width));
        int slot = hash(key, _capacity);

        while (true)
        {
            key_type cur_key = table_keys[slot];
            if (key == cur_key)
            {
                vals[idx * kernel_volume + kernel_idx] = table_vals[slot] - 1; // need to subtract 1 to avoid extra operations in python
            }
            if (table_keys[slot] == EMPTY_CELL)
            {
                return;
            }
            slot = (slot + 1) % _capacity;
        }
    }
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::insert_many_coords(const phi::GPUContext& dev_ctx, const int *coords, const int n){
  insert_coords_kernel<key_type, val_type><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, dev_ctx.stream()>>>(table_keys, table_vals, coords, n, _capacity, _width);
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::insert_coords(const phi::GPUContext& dev_ctx, const phi::DenseTensor& coords){
  insert_many_coords(dev_ctx, coords.data<int>(), coords.dims()[0]);
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::lookup_many_coords(
  const phi::GPUContext& dev_ctx,
  const int* coords, val_type* results,
  const int* kernel_sizes, const int* strides,
  const int n, const int kernel_volume){
  if (kernel_volume % 2)
    lookup_coords_kernel<key_type, val_type, true><<<(n * kernel_volume + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, dev_ctx.stream()>>>(
      table_keys, table_vals, coords, results, kernel_sizes, strides,
      n, _capacity, kernel_volume, _width);
  else
    lookup_coords_kernel<key_type, val_type, false><<<(n * kernel_volume + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, dev_ctx.stream()>>>(
      table_keys, table_vals, coords, results, kernel_sizes, strides,
      n, _capacity, kernel_volume, _width);
}

template <typename key_type, typename val_type>
void GPUHashTable<key_type, val_type>::lookup_coords(
    const phi::GPUContext& dev_ctx,
    const phi::DenseTensor& coords,
    const int* kernel_sizes,
    const int* strides,
    const int kernel_volume,
    phi::DenseTensor* results){
  int32_t* results_data = results->data<int32_t>();
  lookup_many_coords(dev_ctx, coords.data<int>(), results_data, kernel_sizes, strides, coords.dims()[0], kernel_volume);
}

template <typename IntT>
void build_sparse_conv_kmap(
  const phi::GPUContext& dev_ctx,
  const phi::SparseCooTensor& x,
  const std::string& key,
  const std::vector<int>& kernel_sizes,
  const std::vector<int>& strides,
  const int kernel_volume,
  const bool is2D,
  phi::SparseCooTensor* out)
{
  int nnz = x.nnz();
  const phi::KmapCache* in_kmap_cache_ptr = x.GetKmapCache(key);
  out->ClearKmaps();
  phi::KmapCache* out_kmap_cache_ptr = nullptr;
  bool to_insert = false;
  if (in_kmap_cache_ptr == nullptr)
  {
    phi::KmapCache kmap_cache;
    out_kmap_cache_ptr = out->SetKmapCache(key, kmap_cache);
    if (out_kmap_cache_ptr->hashmap_keys == nullptr) {
      phi::DenseTensor* tmp_hashmap_keys = new phi::DenseTensor();
      tmp_hashmap_keys->Resize({2 * x.nnz()});
      dev_ctx.template Alloc<IntT>(tmp_hashmap_keys);
      phi::funcs::SetConstant<phi::GPUContext, IntT> set_zero;
      set_zero(dev_ctx, tmp_hashmap_keys, static_cast<IntT>(0));
      out_kmap_cache_ptr->hashmap_keys = tmp_hashmap_keys;
      to_insert = true;
    }
    if (out_kmap_cache_ptr->hashmap_values == nullptr) {
      phi::DenseTensor* tmp_hashmap_values = new phi::DenseTensor();
      tmp_hashmap_values->Resize({2 * x.nnz()});
      dev_ctx.template Alloc<int32_t>(tmp_hashmap_values);
      phi::funcs::SetConstant<phi::GPUContext, int32_t> set_zero;
      set_zero(dev_ctx, tmp_hashmap_values, static_cast<int32_t>(0));
      out_kmap_cache_ptr->hashmap_values = tmp_hashmap_values;
    }

    if (out_kmap_cache_ptr->coords == nullptr) {
      phi::DenseTensor* tmp_indices = new phi::DenseTensor();
      tmp_indices->Resize({x.indices().dims()[1], x.indices().dims()[0]});
      dev_ctx.template Alloc<int32_t>(tmp_indices);
      // transpose indices
      std::vector<int> perm = {1, 0};
      phi::funcs::TransposeGPUKernelDriver<int32_t>(dev_ctx, x.indices(), perm, tmp_indices);
      out_kmap_cache_ptr->coords = tmp_indices;
    }

    const int divisor = 128;
    const int width = is2D ? 3 : 4;
    auto hashmap = GPUHashTable<IntT, int32_t>(out_kmap_cache_ptr->hashmap_keys, out_kmap_cache_ptr->hashmap_values, divisor, width);
    if (to_insert) {
      hashmap.insert_coords(dev_ctx, *(out_kmap_cache_ptr->coords));
    }

    phi::DenseTensor* tmp_out_in_map = new phi::DenseTensor();
    tmp_out_in_map->Resize({(x.nnz() + divisor - 1) / divisor * divisor, kernel_volume});
    dev_ctx.template Alloc<int32_t>(tmp_out_in_map);
    out_kmap_cache_ptr->out_in_map = tmp_out_in_map;
    phi::funcs::SetConstant<phi::GPUContext, int32_t> set_neg_one;
    set_neg_one(dev_ctx, out_kmap_cache_ptr->out_in_map, static_cast<int32_t>(-1));


    // need to put kernel_sizes and strides to GPU
    auto kernel_sizes_tensor = phi::Empty<int32_t>(dev_ctx, {3});
    phi::TensorFromVector(kernel_sizes, dev_ctx, &kernel_sizes_tensor);
    auto strides_tensor = phi::Empty<int32_t>(dev_ctx, {3});
    phi::TensorFromVector(strides, dev_ctx, &strides_tensor);

    hashmap.lookup_coords(
        dev_ctx, *(out_kmap_cache_ptr->coords), kernel_sizes_tensor.data<int32_t>(), strides_tensor.data<int32_t>(), kernel_volume, out_kmap_cache_ptr->out_in_map);

  } else {
      // out tensor takes the kmaps from x
      out->SetKmaps(x.GetKmaps());
      // force clear the kmaps of x
      const_cast<phi::SparseCooTensor&>(x).ClearKmaps();
  }
  const phi::KmapCache* new_out_kmap_cache_ptr = out->GetKmapCache(key);
  assert(new_out_kmap_cache_ptr != nullptr);
  assert(new_out_kmap_cache_ptr->hashmap_keys != nullptr);
  assert(new_out_kmap_cache_ptr->hashmap_values != nullptr);
  assert(new_out_kmap_cache_ptr->coords != nullptr);
  assert(new_out_kmap_cache_ptr->out_in_map != nullptr);
  return;
}
