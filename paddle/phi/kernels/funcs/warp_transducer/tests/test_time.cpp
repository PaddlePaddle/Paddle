// Copyright 2018-2019, Mingkun Huang
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
  options.batch_first = true;
  options.loc = RNNT_CPU;
  options.num_threads = num_threads;

  size_t cpu_alloc_bytes;
  throw_on_error(get_rnnt_workspace_size(T, L + 1, B, false, &cpu_alloc_bytes),
                 "Error: get_rnnt_workspace_size in run_test");

  std::vector<float> time;
  for (int i = 0; i < 10; ++i) {
    float* grads = new float[len];
    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    start = std::chrono::high_resolution_clock::now();
    throw_on_error(compute_rnnt_loss(acts,
                                     grads,
                                     flat_labels.data(),
                                     label_lengths.data(),
                                     sizes.data(),
                                     A,
                                     B,
                                     costs.data(),
                                     rnnt_cpu_workspace,
                                     options),
                   "Error: compute_rnnt_loss (0) in run_test");
    end = std::chrono::high_resolution_clock::now();

    free(grads);
    free(rnnt_cpu_workspace);
    elapsed = end - start;
    time.push_back(elapsed.count() * 1000);
    std::cout << "compute_rnnt_loss elapsed time: " << elapsed.count() * 1000
              << " ms\n";
  }

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
  cost = cost;

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
