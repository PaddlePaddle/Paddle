// Copyright 2018-2019, Mingkun Huang
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

#include <cmath>
#include <cstdlib>
#include <random>
#include <tuple>
#include <vector>

#include <chrono>

#include <iostream>

#include "test.h"

template <typename T>
void vector_to_gpu(T*& gpu_space, std::vector<T>& vec, gpuStream_t& stream) {
#ifdef __HIPCC__
  hipMalloc(&gpu_space, vec.size() * sizeof(T));
  hipMemcpyAsync(gpu_space,
                 vec.data(),
                 vec.size() * sizeof(T),
                 hipMemcpyHostToDevice,
                 stream);
#else
  cudaMalloc(&gpu_space, vec.size() * sizeof(T));
  cudaMemcpyAsync(gpu_space,
                  vec.data(),
                  vec.size() * sizeof(T),
                  cudaMemcpyHostToDevice,
                  stream);
#endif
}

template <typename T>
void vector_to_gpu(T*& gpu_space,
                   const T* cpu_space,
                   int len,
                   gpuStream_t& stream) {
#ifdef __HIPCC__
  hipMalloc(&gpu_space, len * sizeof(T));
  hipMemcpyAsync(
      gpu_space, cpu_space, len * sizeof(T), hipMemcpyHostToDevice, stream);
#else
  cudaMalloc(&gpu_space, len * sizeof(T));
  cudaMemcpyAsync(
      gpu_space, cpu_space, len * sizeof(T), cudaMemcpyHostToDevice, stream);
#endif
}

bool run_test(int B, int T, int L, int A, int num_threads) {
  std::mt19937 gen(2);

  auto start = std::chrono::high_resolution_clock::now();
  int len = B * T * (L + 1) * A;
  float* acts = genActs(len);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "genActs elapsed time: " << elapsed.count() * 1000 << " ms\n";

  std::vector<std::vector<int>> labels;
  std::vector<int> sizes;

  for (int mb = 0; mb < B; ++mb) {
    labels.push_back(genLabels(A, L));
    sizes.push_back(T);
  }

  std::vector<int> flat_labels;
  std::vector<int> label_lengths;
  for (const auto& l : labels) {
    flat_labels.insert(flat_labels.end(), l.begin(), l.end());
    label_lengths.push_back(l.size());
  }

  std::vector<float> costs(B);

  rnntOptions options{};
  options.maxT = T;
  options.maxU = L + 1;
  options.blank_label = 0;
  options.loc = RNNT_GPU;
  gpuStream_t stream;
#ifdef __HIPCC__
  hipStreamCreate(&stream);
#else
  cudaStreamCreate(&stream);
#endif
  options.stream = stream;
  options.num_threads = num_threads;

  float* acts_gpu;
  vector_to_gpu<float>(acts_gpu, acts, len, stream);
  // cudaMalloc(&acts_gpu, len * sizeof(float));
  // cudaMemcpyAsync(acts_gpu, acts, len * sizeof(float),
  // cudaMemcpyHostToDevice, stream);
  float* grads_gpu;
#ifdef __HIPCC__
  hipMalloc(&grads_gpu, len * sizeof(float));
#else
  cudaMalloc(&grads_gpu, len * sizeof(float));
#endif
  int* label_gpu;
  vector_to_gpu(label_gpu, flat_labels, stream);
  // cudaMalloc(&label_gpu, flat_labels.size() * sizeof(int))
  // cudaMemcpyAsync(label_gpu, flat_labels.data(), flat_labels.size() *
  // sizeof(int), cudaMemcpyHostToDevice, stream);
  int* label_length_gpu;
  vector_to_gpu(label_length_gpu, label_lengths, stream);
  // cudaMalloc(&label_length_gpu, label_lengths.size() * sizeof(int));
  // cudaMemcpyAsync(label_length_gpu, label_lengths.data(),
  // label_lengths.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
  int* input_length_gpu;
  vector_to_gpu(input_length_gpu, sizes, stream);
  // cudaMalloc(&input_length_gpu, sizes.size() * sizeof(int));
  // cudaMemcpyAsync(input_length_gpu, sizes.data(), sizes.size() * sizeof(int),
  // cudaMemcpyHostToDevice, stream);

  size_t gpu_alloc_bytes;
  throw_on_error(get_rnnt_workspace_size(T, L + 1, B, true, &gpu_alloc_bytes),
                 "Error: get_rnnt_workspace_size in run_test");

  std::vector<float> time;
  for (int i = 0; i < 10; ++i) {
    void* rnnt_gpu_workspace;
#ifdef __HIPCC__
    hipMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#else
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#endif

    start = std::chrono::high_resolution_clock::now();
    throw_on_error(compute_rnnt_loss(acts_gpu,
                                     grads_gpu,
                                     label_gpu,
                                     label_length_gpu,
                                     input_length_gpu,
                                     A,
                                     B,
                                     costs.data(),
                                     rnnt_gpu_workspace,
                                     options),
                   "Error: compute_rnnt_loss (0) in run_test");
    end = std::chrono::high_resolution_clock::now();

#ifdef __HIPCC__
    hipFree(rnnt_gpu_workspace);
#else
    cudaFree(rnnt_gpu_workspace);
#endif
    elapsed = end - start;
    time.push_back(elapsed.count() * 1000);
    std::cout << "compute_rnnt_loss elapsed time: " << elapsed.count() * 1000
              << " ms\n";
  }

#ifdef __HIPCC__
  hipFree(grads_gpu);
  hipFree(label_gpu);
  hipFree(label_length_gpu);
  hipFree(input_length_gpu);
#else
  cudaFree(grads_gpu);
  cudaFree(label_gpu);
  cudaFree(label_length_gpu);
  cudaFree(input_length_gpu);
#endif

  float sum = 0;
  for (int i = 0; i < 10; ++i) {
    sum += time[i];
  }
  sum /= time.size();

  float std = 0;
  for (int i = 0; i < 10; ++i) {
    std += (time[i] - sum) * (time[i] - sum);
  }
  std /= time.size();

  std::cout << "average 10 time cost: " << sum << " ms variance: " << std
            << std::endl;

  float cost = std::accumulate(costs.begin(), costs.end(), 0.);

  free(acts);
  return true;
}

int main(int argc, char** argv) {
  if (argc == 1) {
    (void)0;
  } else if (argc < 5) {
    std::cerr << "Arguments: <Batch size> <Time step> <Label length> <Alphabet "
                 "size>\n";
    return 1;
  }

  int B = 0;
  int T = 0;
  int L = 0;
  int A = 0;
  if (argc == 5) {
    B = atoi(argv[1]);
    T = atoi(argv[2]);
    L = atoi(argv[3]);
    A = atoi(argv[4]);
  } else {
    B = 32;
    T = 50;
    L = 20;
    A = 50;
  }

  std::cout << "Arguments: "
            << "\nBatch size: " << B << "\nTime step: " << T
            << "\nLabel length: " << L << "\nAlphabet size: " << A << std::endl;

  int num_threads = 1;
  if (argc >= 6) {
    num_threads = atoi(argv[5]);
    std::cout << "Num threads: " << num_threads << std::endl;
  }

  run_test(B, T, L, A, num_threads);
}
