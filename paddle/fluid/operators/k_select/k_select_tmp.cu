#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "k_select.h"

#define DIVUP(x, y) (((x)+(y)-1)/(y))

#define BLOCK_SIZE 256
#define MAX_BLOCK 128

template<typename T>
__global__ void null_rw_kernel(T* input, T* output, int count) {
  int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  
  T tmp;
  for (int i = tid; i < count; i += MAX_BLOCK * BLOCK_SIZE) {
    tmp = input[i];
    output[i] = tmp;
  }
}

template<typename T>
__global__ void get_topk_kernel(void* encode, T* input, int k, int count) {
  int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  
  int* idx = static_cast<int*>(encode);
  T* val = reinterpret_cast<T*>(&idx[k]);

  for (int i = tid; i < k; i += MAX_BLOCK * BLOCK_SIZE) {
    idx[i] = i;
    val[i] = input[i];   
  }
}

bool k_select(float* input, int input_count, void* encode, int k, cudaStream_t stream) {
  assert(input != NULL);
  assert(encode != NULL);
  assert(input_count > 0 && k > 0 && input_count > k);
  int blocks = min(MAX_BLOCK, DIVUP(k, BLOCK_SIZE));
  get_topk_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(encode, input, k, input_count); 

  // null read
  blocks = min(MAX_BLOCK, DIVUP(input_count, BLOCK_SIZE));
  null_rw_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, input, input_count);
  return true;
}

