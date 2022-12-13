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
#include <random>
#include <tuple>
#include <vector>

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

bool small_test() {
  const int B = 1;
  const int alphabet_size = 5;
  const int T = 2;
  const int U = 3;

  std::vector<float> acts = {0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1,
                             0.1, 0.1, 0.2, 0.8, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1,
                             0.1, 0.1, 0.2, 0.1, 0.1, 0.7, 0.1, 0.2, 0.1, 0.1};
  // std::vector<float> log_probs(acts.size());
  // softmax(acts.data(), alphabet_size, B * T * U, log_probs.data(), true);

  float expected_score = 4.495666;

  std::vector<int> labels = {1, 2};
  std::vector<int> label_lengths = {2};

  std::vector<int> lengths;
  lengths.push_back(T);

  float score;

  rnntOptions options{};
  options.maxT = T;
  options.maxU = U;
  options.loc = RNNT_GPU;
  options.blank_label = 0;
  gpuStream_t stream;
#ifdef __HIPCC__
  hipStreamCreate(&stream);
#else
  cudaStreamCreate(&stream);
#endif
  options.stream = stream;
  options.num_threads = 1;

  float* acts_gpu;
  vector_to_gpu(acts_gpu, acts, stream);
  int* label_gpu;
  vector_to_gpu(label_gpu, labels, stream);
  int* label_length_gpu;
  vector_to_gpu(label_length_gpu, label_lengths, stream);
  int* input_length_gpu;
  vector_to_gpu(input_length_gpu, lengths, stream);

  size_t gpu_alloc_bytes;
  throw_on_error(get_rnnt_workspace_size(T, U, B, true, &gpu_alloc_bytes),
                 "Error: get_rnnt_workspace_size in small_test");

  void* rnnt_gpu_workspace;
#ifdef __HIPCC__
  hipMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#else
  cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#endif

  throw_on_error(compute_rnnt_loss(acts_gpu,
                                   NULL,
                                   label_gpu,
                                   label_length_gpu,
                                   input_length_gpu,
                                   alphabet_size,
                                   lengths.size(),
                                   &score,
                                   rnnt_gpu_workspace,
                                   options),
                 "Error: compute_rnnt_loss in small_test");

#ifdef __HIPCC__
  hipFree(rnnt_gpu_workspace);
  hipFree(acts_gpu);
  hipFree(label_gpu);
  hipFree(label_length_gpu);
  hipFree(input_length_gpu);
#else
  cudaFree(rnnt_gpu_workspace);
  cudaFree(acts_gpu);
  cudaFree(label_gpu);
  cudaFree(label_length_gpu);
  cudaFree(input_length_gpu);
#endif

  const float eps = 1e-4;

  const float lb = expected_score - eps;
  const float ub = expected_score + eps;

  return (score > lb && score < ub);
}

bool options_test() {
  const int alphabet_size = 3;
  const int T = 4;
  const int L = 3;
  const int minibatch = 2;

  std::vector<float> acts = {
      0.065357, 0.787530, 0.081592, 0.529716, 0.750675, 0.754135, 0.609764,
      0.868140, 0.622532, 0.668522, 0.858039, 0.164539, 0.989780, 0.944298,
      0.603168, 0.946783, 0.666203, 0.286882, 0.094184, 0.366674, 0.736168,
      0.166680, 0.714154, 0.399400, 0.535982, 0.291821, 0.612642, 0.324241,
      0.800764, 0.524106, 0.779195, 0.183314, 0.113745, 0.240222, 0.339470,
      0.134160, 0.505562, 0.051597, 0.640290, 0.430733, 0.829473, 0.177467,
      0.320700, 0.042883, 0.302803, 0.675178, 0.569537, 0.558474, 0.083132,
      0.060165, 0.107958, 0.748615, 0.943918, 0.486356, 0.418199, 0.652408,
      0.024243, 0.134582, 0.366342, 0.295830, 0.923670, 0.689929, 0.741898,
      0.250005, 0.603430, 0.987289, 0.592606, 0.884672, 0.543450, 0.660770,
      0.377128, 0.358021};
  // std::vector<float> log_probs(acts.size());
  // softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(),
  // true);

  std::vector<float> expected_grads = {
      -0.186844, -0.062555, 0.249399,  -0.203377, 0.202399,  0.000977,
      -0.141016, 0.079123,  0.061893,  -0.011552, -0.081280, 0.092832,
      -0.154257, 0.229433,  -0.075176, -0.246593, 0.146405,  0.100188,
      -0.012918, -0.061593, 0.074512,  -0.055986, 0.219831,  -0.163845,
      -0.497627, 0.209240,  0.288387,  0.013605,  -0.030220, 0.016615,
      0.113925,  0.062781,  -0.176706, -0.667078, 0.367659,  0.299419,
      -0.356344, -0.055347, 0.411691,  -0.096922, 0.029459,  0.067463,
      -0.063518, 0.027654,  0.035863,  -0.154499, -0.073942, 0.228441,
      -0.166790, -0.000088, 0.166878,  -0.172370, 0.105565,  0.066804,
      0.023875,  -0.118256, 0.094381,  -0.104707, -0.108934, 0.213642,
      -0.369844, 0.180118,  0.189726,  0.025714,  -0.079462, 0.053748,
      0.122328,  -0.238789, 0.116460,  -0.598687, 0.302203,  0.296484};

  // Calculate the expected scores analytically
  std::vector<double> expected_scores(2);
  expected_scores[0] = 4.2806528590890736;
  expected_scores[1] = 3.9384369822503591;

  std::vector<int> labels = {1, 2, 1, 1};

  std::vector<int> label_lengths = {2, 2};

  std::vector<int> lengths = {4, 4};

  std::vector<float> grads(acts.size());
  std::vector<float> scores(2);

  rnntOptions options{};
  options.maxT = T;
  options.maxU = L;
  options.loc = RNNT_GPU;
  gpuStream_t stream;
#ifdef __HIPCC__
  hipStreamCreate(&stream);
#else
  cudaStreamCreate(&stream);
#endif
  options.stream = stream;
  options.num_threads = 1;

  float* acts_gpu;
  vector_to_gpu(acts_gpu, acts, stream);
  float* grads_gpu;
#ifdef __HIPCC__
  hipMalloc(&grads_gpu, grads.size() * sizeof(float));
#else
  cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
#endif
  int* label_gpu;
  vector_to_gpu(label_gpu, labels, stream);
  int* label_length_gpu;
  vector_to_gpu(label_length_gpu, label_lengths, stream);
  int* input_length_gpu;
  vector_to_gpu(input_length_gpu, lengths, stream);

  size_t gpu_alloc_bytes;
  throw_on_error(
      get_rnnt_workspace_size(T, L, minibatch, true, &gpu_alloc_bytes),
      "Error: get_rnnt_workspace_size in options_test");

  void* rnnt_gpu_workspace;
#ifdef __HIPCC__
  hipMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#else
  cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#endif

  throw_on_error(compute_rnnt_loss(acts_gpu,
                                   grads_gpu,
                                   label_gpu,
                                   label_length_gpu,
                                   input_length_gpu,
                                   alphabet_size,
                                   lengths.size(),
                                   scores.data(),
                                   rnnt_gpu_workspace,
                                   options),
                 "Error: compute_rnnt_loss in small_test");

#ifdef __HIPCC__
  hipMemcpyAsync(grads.data(),
                 grads_gpu,
                 grads.size() * sizeof(float),
                 hipMemcpyDeviceToHost,
                 stream);

  hipFree(rnnt_gpu_workspace);
  hipFree(acts_gpu);
  hipFree(grads_gpu);
  hipFree(label_gpu);
  hipFree(label_length_gpu);
  hipFree(input_length_gpu);
#else
  cudaMemcpyAsync(grads.data(),
                  grads_gpu,
                  grads.size() * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream);

  cudaFree(rnnt_gpu_workspace);
  cudaFree(acts_gpu);
  cudaFree(grads_gpu);
  cudaFree(label_gpu);
  cudaFree(label_length_gpu);
  cudaFree(input_length_gpu);
#endif

  const double eps = 1e-4;

  bool result = true;
  // activations gradient check
  for (int i = 0; i < grads.size(); i++) {
    const double lb = expected_grads[i] - eps;
    const double ub = expected_grads[i] + eps;
    if (!(grads[i] > lb && grads[i] < ub)) {
      std::cerr << "grad mismatch in options_test"
                << " expected grad: " << expected_grads[i]
                << " calculated score: " << grads[i] << " !(" << lb << " < "
                << grads[i] << " < " << ub << ")" << std::endl;
      result = false;
    }
  }

  for (int i = 0; i < 2; i++) {
    const double lb = expected_scores[i] - eps;
    const double ub = expected_scores[i] + eps;
    if (!(scores[i] > lb && scores[i] < ub)) {
      std::cerr << "score mismatch in options_test"
                << " expected score: " << expected_scores[i]
                << " calculated score: " << scores[i] << " !(" << lb << " < "
                << scores[i] << " < " << ub << ")" << std::endl;
      result = false;
    }
  }
  return result;
}

bool inf_test() {
  const int alphabet_size = 15;
  const int T = 50;
  const int L = 10;
  const int minibatch = 1;

  std::vector<int> labels = genLabels(alphabet_size, L - 1);
  labels[0] = 2;
  std::vector<int> label_lengths = {L - 1};

  std::vector<float> acts(alphabet_size * T * L * minibatch);
  genActs(acts);

  // std::vector<float> log_probs(acts.size());
  // softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(),
  // true);

  std::vector<int> sizes;
  sizes.push_back(T);

  std::vector<float> grads(acts.size());

  float cost;

  rnntOptions options{};
  options.maxT = T;
  options.maxU = L;
  options.loc = RNNT_GPU;
  gpuStream_t stream;
#ifdef __HIPCC__
  hipStreamCreate(&stream);
#else
  cudaStreamCreate(&stream);
#endif
  options.stream = stream;
  options.num_threads = 1;

  float* acts_gpu;
  vector_to_gpu(acts_gpu, acts, stream);
  float* grads_gpu;
#ifdef __HIPCC__
  hipMalloc(&grads_gpu, grads.size() * sizeof(float));
#else
  cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
#endif
  int* label_gpu;
  vector_to_gpu(label_gpu, labels, stream);
  int* label_length_gpu;
  vector_to_gpu(label_length_gpu, label_lengths, stream);
  int* input_length_gpu;
  vector_to_gpu(input_length_gpu, sizes, stream);

  size_t gpu_alloc_bytes;
  throw_on_error(
      get_rnnt_workspace_size(T, L, minibatch, true, &gpu_alloc_bytes),
      "Error: get_rnnt_workspace_size in inf_test");

  void* rnnt_gpu_workspace;
#ifdef __HIPCC__
  hipMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#else
  cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#endif

  throw_on_error(compute_rnnt_loss(acts_gpu,
                                   grads_gpu,
                                   label_gpu,
                                   label_length_gpu,
                                   input_length_gpu,
                                   alphabet_size,
                                   sizes.size(),
                                   &cost,
                                   rnnt_gpu_workspace,
                                   options),
                 "Error: compute_rnnt_loss in small_test");

#ifdef __HIPCC__
  hipMemcpyAsync(grads.data(),
                 grads_gpu,
                 grads.size() * sizeof(float),
                 hipMemcpyDeviceToHost,
                 stream);

  hipFree(rnnt_gpu_workspace);
  hipFree(acts_gpu);
  hipFree(grads_gpu);
  hipFree(label_gpu);
  hipFree(label_length_gpu);
  hipFree(input_length_gpu);
#else
  cudaMemcpyAsync(grads.data(),
                  grads_gpu,
                  grads.size() * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream);

  cudaFree(rnnt_gpu_workspace);
  cudaFree(acts_gpu);
  cudaFree(grads_gpu);
  cudaFree(label_gpu);
  cudaFree(label_length_gpu);
  cudaFree(input_length_gpu);
#endif

  bool status = true;
  status &= !std::isinf(cost);

  for (int i = 0; i < alphabet_size * L * T * minibatch; ++i)
    status &= !std::isnan(grads[i]);

  return status;
}

void numeric_grad(float* acts,
                  int* flat_labels,
                  int* label_lengths,
                  int* sizes,
                  int alphabet_size,
                  int minibatch,
                  void* rnnt_gpu_workspace,
                  rnntOptions& options,
                  std::vector<float>& num_grad) {
  float epsilon = 1e-2;
  float act;

  for (int i = 0; i < num_grad.size(); ++i) {
    std::vector<float> costsP1(minibatch);
    std::vector<float> costsP2(minibatch);

#ifdef __HIPCC__
    hipMemcpy(&act, &acts[i], sizeof(float), hipMemcpyDeviceToHost);
#else
    cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
#endif
    act += epsilon;
#ifdef __HIPCC__
    hipMemcpy(&acts[i], &act, sizeof(float), hipMemcpyHostToDevice);
#else
    cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
#endif
    throw_on_error(compute_rnnt_loss(acts,
                                     NULL,
                                     flat_labels,
                                     label_lengths,
                                     sizes,
                                     alphabet_size,
                                     minibatch,
                                     costsP1.data(),
                                     rnnt_gpu_workspace,
                                     options),
                   "Error: compute_rnnt_loss (1) in grad_check");

#ifdef __HIPCC__
    hipMemcpy(&act, &acts[i], sizeof(float), hipMemcpyDeviceToHost);
#else
    cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
#endif
    act -= 2 * epsilon;
#ifdef __HIPCC__
    hipMemcpy(&acts[i], &act, sizeof(float), hipMemcpyHostToDevice);
#else
    cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
#endif
    throw_on_error(compute_rnnt_loss(acts,
                                     NULL,
                                     flat_labels,
                                     label_lengths,
                                     sizes,
                                     alphabet_size,
                                     minibatch,
                                     costsP2.data(),
                                     rnnt_gpu_workspace,
                                     options),
                   "Error: compute_rnnt_loss (2) in grad_check");

    float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.);
    float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.);
#ifdef __HIPCC__
    hipMemcpy(&act, &acts[i], sizeof(float), hipMemcpyDeviceToHost);
#else
    cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
#endif
    act += epsilon;
#ifdef __HIPCC__
    hipMemcpy(&acts[i], &act, sizeof(float), hipMemcpyHostToDevice);
#else
    cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
#endif
    num_grad[i] = (costP1 - costP2) / (2 * epsilon);
  }
}

bool grad_check(int T,
                int L,
                int alphabet_size,
                std::vector<float>& acts,
                const std::vector<std::vector<int>>& labels,
                std::vector<int>& sizes,
                float tol) {
  const int minibatch = labels.size();

  std::vector<int> flat_labels;
  std::vector<int> label_lengths;
  for (const auto& l : labels) {
    flat_labels.insert(flat_labels.end(), l.begin(), l.end());
    label_lengths.push_back(l.size());
  }

  std::vector<float> costs(minibatch);

  std::vector<float> grads(acts.size());

  rnntOptions options{};
  options.maxT = T;
  options.maxU = L;
  options.loc = RNNT_GPU;
  gpuStream_t stream;
#ifdef __HIPCC__
  hipStreamCreate(&stream);
#else
  cudaStreamCreate(&stream);
#endif
  options.stream = stream;
  options.num_threads = 1;

  float* acts_gpu;
  vector_to_gpu(acts_gpu, acts, stream);
  float* grads_gpu;
#ifdef __HIPCC__
  hipMalloc(&grads_gpu, grads.size() * sizeof(float));
#else
  cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
#endif
  int* label_gpu;
  vector_to_gpu(label_gpu, flat_labels, stream);
  int* label_length_gpu;
  vector_to_gpu(label_length_gpu, label_lengths, stream);
  int* input_length_gpu;
  vector_to_gpu(input_length_gpu, sizes, stream);
  options.num_threads = 1;

  size_t gpu_alloc_bytes;
  throw_on_error(
      get_rnnt_workspace_size(T, L, sizes.size(), true, &gpu_alloc_bytes),
      "Error: get_rnnt_workspace_size in grad_check");

  void* rnnt_gpu_workspace;
#ifdef __HIPCC__
  hipMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#else
  cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);
#endif

  throw_on_error(compute_rnnt_loss(acts_gpu,
                                   grads_gpu,
                                   label_gpu,
                                   label_length_gpu,
                                   input_length_gpu,
                                   alphabet_size,
                                   sizes.size(),
                                   costs.data(),
                                   rnnt_gpu_workspace,
                                   options),
                 "Error: compute_rnnt_loss (0) in grad_check");

  float cost = std::accumulate(costs.begin(), costs.end(), 0.);
#ifdef __HIPCC__
  hipMemcpyAsync(grads.data(),
                 grads_gpu,
                 grads.size() * sizeof(float),
                 hipMemcpyDeviceToHost,
                 stream);
#else
  cudaMemcpyAsync(grads.data(),
                  grads_gpu,
                  grads.size() * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream);
#endif

  std::vector<float> num_grad(grads.size());

  // perform 2nd order central differencing
  numeric_grad(acts_gpu,
               label_gpu,
               label_length_gpu,
               input_length_gpu,
               alphabet_size,
               minibatch,
               rnnt_gpu_workspace,
               options,
               num_grad);

#ifdef __HIPCC__
  hipFree(acts_gpu);
  hipFree(rnnt_gpu_workspace);
  hipFree(grads_gpu);
  hipFree(label_gpu);
  hipFree(label_length_gpu);
  hipFree(input_length_gpu);
#else
  cudaFree(acts_gpu);
  cudaFree(rnnt_gpu_workspace);
  cudaFree(grads_gpu);
  cudaFree(label_gpu);
  cudaFree(label_length_gpu);
  cudaFree(input_length_gpu);
#endif

  float diff = rel_diff(grads, num_grad);

  return diff < tol;
}

bool run_tests() {
  std::vector<std::tuple<int, int, int, int, float>> problem_sizes = {
      std::make_tuple(20, 50, 15, 1, 1e-2),
      std::make_tuple(5, 10, 5, 65, 1e-2)};

  std::mt19937 gen(2);

  bool status = true;
  for (auto problem : problem_sizes) {
    int alphabet_size, T, L, minibatch;
    float tol;
    std::tie(alphabet_size, T, L, minibatch, tol) = problem;

    std::vector<float> acts(alphabet_size * T * L * minibatch);
    genActs(acts);

    std::vector<float> log_probs(acts.size());
    softmax(
        acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

    std::vector<std::vector<int>> labels;
    std::vector<int> sizes;
    for (int mb = 0; mb < minibatch; ++mb) {
      int actual_length = L - 1;
      labels.push_back(genLabels(alphabet_size, actual_length));
      sizes.push_back(T);
    }

    status &= grad_check(T, L, alphabet_size, acts, labels, sizes, tol);
  }

  return status;
}

int main(void) {
  if (get_warprnnt_version() != 1) {
    std::cerr << "Invalid Warp-transducer version." << std::endl;
    return 1;
  }

  std::cout << "Running gpu tests" << std::endl;

  bool status = true;
  status &= small_test();
  printf("finish small_test %d\n", status);
  status &= options_test();
  printf("finish options_test %d\n", status);
  status &= inf_test();
  printf("finish inf_test %d\n", status);
  status &= run_tests();
  printf("finished %d\n", status);

  if (status) {
    std::cout << "Tests pass" << std::endl;
    return 0;
  } else {
    std::cout << "Some or all tests fail" << std::endl;
    return 1;
  }
}
