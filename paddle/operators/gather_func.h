/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <cstring>
#include "paddle/framework/tensor.h"
#include "paddle/platform/place.h"
#include "paddle/framework/ddim.h"

/**
 * Return a new tensor from source tensor, gathered according to index
 * input[src]: type-T source Tensor
 * input[Index]: type-int index Tensor (1-D)
 * return: output tensor
 */
template <typename place, typename T>
Tensor* Gather_func(Tensor* Src, Tensor* Index) {
	// assert index is an int-type tensor?
	// assert(Index->istype(int));

	// check index of shape 1-D
	assert(Index->dims().size()==1);
	int index_size = Index->dims()[0];

	// Source shape
	auto src_dims = Src->dims();
	DDim output_dims(dims_src);
	// Create a tensor of shape [index_size, dim_src[1:]]
	output_dims[0] = index_size;

	Tensor* New_tensor;
	float* output = nullptr;

	/* slice size */
	int slice_size = 1;
	for(unsigned int i = 0; i < src_dims.size(); ++i)
		slice_size *= src_dims[i];

	/* Gathering */
	if (place == CPUPlace()) {
		// init for CPU
		output = New_tensor.mutable_data<T>(output_dims, CPUPlace());
		CPUGather(Src->data(), Index->data(), slice_size, new_tensor->mutable_data());
	} else { // GPU
		// init for GPU
		output = New_tensor.mutable_data<T>(output_dims, GPUPlace());
		/* how to specialize device??*/
		GPUGather(d, Src->data(), Index->data(), slice_size, new_tensor->mutable_data());
	}
	return New_tensor;
}

/* Implementation of CPU copy */
template<typename T>
void CPUGather(const T* params, const int* indices, 
			   const int slice_size, const int index_size,
			   T* output) {
  const size_t slice_bytes = slice_size * sizeof(T);

  for(int i = 0; i < index_size; ++i)
  	int index_ = indices[i];
  	/* copy src[index_] to output[i] */
  	memcpy(output + i * slice_bytes,
  		params + index_ * slice_bytes,
  		slice_bytes);
}

/* Implementation of GPU copy:
   I suppose the GPUDevice& d, contains gpu_id and thread_id
   d = cuda_stream(gpu_id_, stream_id_);
*/
template<typename T>
void GPUGather(const GPUDevice& d,
			   const T* src, const int* Index, 
	           const int slice_size, const int index_size,
	           T* output) {
	int block_count = slice_size * index_size;
	int thread_per_block = 1024;

	GatherOpKernel<T>
          <<<block_count, thread_per_block, 0, d.stream()>>>(
              src, Index, output, slice_size,
              indices_size, slice_size, out_size);
}

template <typename T>
__global__ void GatherOpKernel(const T* params, const int* indices, T* out,
                               int64 indices_size,
                               int64 slice_size, int64 out_size) {
  /* I suppose we have the following macro, 
     which I strongly suggest that we should put in cuda:
  #define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)
  */
  CUDA_1D_KERNEL_LOOP(i, out_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size; // offset inside the slice
    int gather_i = indices[indices_i];
    int params_i = gather_i * slice_size + slice_i;
    out[i] = *(params + params_i);
  } 
}
