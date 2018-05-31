/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <gtest/gtest.h>
#include <unordered_map>

#include "paddle/fluid/memory/detail/memory_block.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/memory/memory.h"

#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

// This unit test is an example comparing the performance between using pinned
// memory and not. In general, using pinned memory will be faster.
template <typename T>
__global__ void Kernel(T* output, int dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < dim) {
    output[tid] = output[tid] * output[tid] / 100;
  }
}

template <typename Place>
float test_pinned_memory() {
  Place cpu_place;
  paddle::platform::CUDAPlace cuda_place;

  const int data_size = 4096;
  const int iteration = 10;

  // create event start and end
  cudaEvent_t start_e, stop_e, copying_e;
  float elapsedTime = 0;
  cudaEventCreate(&start_e);
  cudaEventCreate(&stop_e);
  cudaEventCreate(&copying_e);

  // create computation stream, data copying stream
  cudaStream_t computation_stream, copying_stream;
  cudaStreamCreate(&computation_stream);
  cudaStreamCreate(&copying_stream);

  // create record event, pinned memory, gpu memory
  std::vector<cudaEvent_t> record_event(iteration);
  std::vector<float*> input_pinned_mem(iteration);
  std::vector<float*> gpu_mem(iteration);
  std::vector<float*> output_pinned_mem(iteration);

  // initial data
  for (int j = 0; j < iteration; ++j) {
    cudaEventCreateWithFlags(&record_event[j], cudaEventDisableTiming);
    cudaEventCreate(&(record_event[j]));
    input_pinned_mem[j] = static_cast<float*>(
        paddle::memory::Alloc(cpu_place, data_size * sizeof(float)));
    output_pinned_mem[j] = static_cast<float*>(
        paddle::memory::Alloc(cpu_place, data_size * sizeof(float)));
    gpu_mem[j] = static_cast<float*>(
        paddle::memory::Alloc(cuda_place, data_size * sizeof(float)));

    for (int k = 0; k < data_size; ++k) {
      input_pinned_mem[j][k] = k;
    }
  }

  cudaEventRecord(start_e, computation_stream);

  // computation
  for (int m = 0; m < 30; ++m) {
    for (int i = 0; i < iteration; ++i) {
      // cpu -> GPU on computation stream.
      // note: this operation is async for pinned memory.
      paddle::memory::Copy(cuda_place, gpu_mem[i], cpu_place,
                           input_pinned_mem[i], data_size * sizeof(float),
                           computation_stream);

      // call kernel on computation stream.
      Kernel<<<4, 1024, 0, computation_stream>>>(gpu_mem[i], data_size);

      // record event_computation on computation stream
      cudaEventRecord(record_event[i], computation_stream);

      // wait event_computation on copy stream.
      // note: this operation is async.
      cudaStreamWaitEvent(copying_stream, record_event[i], 0);

      // copy data GPU->CPU, on copy stream.
      // note: this operation is async for pinned memory.
      paddle::memory::Copy(cpu_place, output_pinned_mem[i], cuda_place,
                           gpu_mem[i], data_size * sizeof(float),
                           copying_stream);
    }
  }

  cudaEventRecord(copying_e, copying_stream);
  cudaStreamWaitEvent(computation_stream, copying_e, 0);

  cudaEventRecord(stop_e, computation_stream);

  cudaEventSynchronize(start_e);
  cudaEventSynchronize(stop_e);
  cudaEventElapsedTime(&elapsedTime, start_e, stop_e);

  // std::cout << cpu_place << " "
  //          << "time consume:" << elapsedTime / 30 << std::endl;

  for (int l = 0; l < iteration; ++l) {
    for (int k = 0; k < data_size; ++k) {
      float temp = input_pinned_mem[l][k];
      temp = temp * temp / 100;
      EXPECT_FLOAT_EQ(temp, output_pinned_mem[l][k]);
    }
  }

  // destroy resource
  cudaEventDestroy(copying_e);
  cudaEventDestroy(start_e);
  cudaEventDestroy(stop_e);
  for (int j = 0; j < 10; ++j) {
    cudaEventDestroy((record_event[j]));
    paddle::memory::Free(cpu_place, input_pinned_mem[j]);
    paddle::memory::Free(cpu_place, output_pinned_mem[j]);
    paddle::memory::Free(cuda_place, gpu_mem[j]);
  }
  return elapsedTime / 30;
}

TEST(CPUANDCUDAPinned, CPUAllocatorAndCUDAPinnedAllocator) {
  // Generally speaking, operation on pinned_memory is faster than that on
  // unpinned-memory, but if this unit test fails frequently, please close this
  // test for the time being.
  float time1 = test_pinned_memory<paddle::platform::CPUPlace>();
  float time2 = test_pinned_memory<paddle::platform::CUDAPinnedPlace>();
  EXPECT_GT(time1, time2);
}
