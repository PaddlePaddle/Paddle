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
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

#include "test.h"

bool small_test() {
  const int B = 1;
  const int alphabet_size = 5;
  const int T = 2;
  const int U = 3;

  std::vector<float> acts = {0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1,
                             0.1, 0.1, 0.2, 0.8, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1,
                             0.1, 0.1, 0.2, 0.1, 0.1, 0.7, 0.1, 0.2, 0.1, 0.1};
  std::vector<float> log_probs(acts.size());
  softmax(acts.data(), alphabet_size, B * T * U, log_probs.data(), true);

  float expected_score = 4.495666;

  std::vector<int> labels = {1, 2};
  std::vector<int> label_lengths = {2};

  std::vector<int> lengths;
  lengths.push_back(T);

  float score;

  rnntOptions options{};
  options.maxT = T;
  options.maxU = U;
  options.loc = RNNT_CPU;
  options.batch_first = true;
  options.blank_label = 0;
  options.num_threads = 1;

  size_t cpu_alloc_bytes;
  throw_on_error(get_rnnt_workspace_size(T, U, B, false, &cpu_alloc_bytes),
                 "Error: get_rnnt_workspace_size in small_test");

  void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

  throw_on_error(compute_rnnt_loss(log_probs.data(),
                                   NULL,
                                   labels.data(),
                                   label_lengths.data(),
                                   lengths.data(),
                                   alphabet_size,
                                   lengths.size(),
                                   &score,
                                   rnnt_cpu_workspace,
                                   options),
                 "Error: compute_rnnt_loss in small_test");

  free(rnnt_cpu_workspace);
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
  std::vector<float> log_probs(acts.size());
  softmax(
      acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

  std::vector<float> expected_grads = {
      -0.432226, -0.567774, 0,         -0.365650, 0,         -0.202123,
      -0.202123, 0,         0,         -0.165217, -0.267010, 0,
      -0.394365, 0,         -0.238294, -0.440418, 0,         0,
      -0.052130, -0.113087, 0,         -0.183138, 0,         -0.324314,
      -0.764732, 0,         0,         0,         -0.052130, 0,
      0,         0,         -0.235268, -1,        0,         0,
      -0.716142, -0.283858, 0,         -0.183829, -0.100028, 0,
      -0.100028, 0,         0,         -0.411218, -0.304924, 0,
      -0.329576, -0.159178, 0,         -0.259206, 0,         0,
      -0.116076, -0.295142, 0,         -0.286533, -0.338184, 0,
      -0.597390, 0,         0,         0,         -0.116076, 0,
      0,         -0.402610, 0,         -1,        0,         0};
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
  options.loc = RNNT_CPU;
  options.num_threads = 1;
  options.batch_first = true;

  size_t cpu_alloc_bytes;
  throw_on_error(
      get_rnnt_workspace_size(T, L, minibatch, false, &cpu_alloc_bytes),
      "Error: get_rnnt_workspace_size in options_test");

  void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

  throw_on_error(compute_rnnt_loss(log_probs.data(),
                                   grads.data(),
                                   labels.data(),
                                   label_lengths.data(),
                                   lengths.data(),
                                   alphabet_size,
                                   lengths.size(),
                                   scores.data(),
                                   rnnt_cpu_workspace,
                                   options),
                 "Error: compute_rnnt_loss in options_test");

  free(rnnt_cpu_workspace);

  const double eps = 1e-4;

  bool result = true;
  // activations gradient check
  for (size_t i = 0; i < grads.size(); i++) {
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

  std::vector<float> log_probs(acts.size());
  softmax(
      acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

  std::vector<int> sizes;
  sizes.push_back(T);

  std::vector<float> grads(acts.size());

  float cost;

  rnntOptions options{};
  options.maxT = T;
  options.maxU = L;
  options.loc = RNNT_CPU;
  options.num_threads = 1;
  options.batch_first = true;

  size_t cpu_alloc_bytes;
  throw_on_error(
      get_rnnt_workspace_size(T, L, minibatch, false, &cpu_alloc_bytes),
      "Error: get_rnnt_workspace_size in inf_test");

  void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

  throw_on_error(compute_rnnt_loss(acts.data(),
                                   grads.data(),
                                   labels.data(),
                                   label_lengths.data(),
                                   sizes.data(),
                                   alphabet_size,
                                   sizes.size(),
                                   &cost,
                                   rnnt_cpu_workspace,
                                   options),
                 "Error: compute_rnnt_loss in inf_test");

  free(rnnt_cpu_workspace);

  bool status = true;
  status &= !std::isinf(cost);

  for (int i = 0; i < alphabet_size * L * T * minibatch; ++i)
    status &= !std::isnan(grads[i]);

  return status;
}

float numeric_grad(std::vector<float>& acts,
                   std::vector<int>& flat_labels,
                   std::vector<int>& label_lengths,
                   std::vector<int> sizes,
                   int alphabet_size,
                   int minibatch,
                   void* rnnt_cpu_workspace,
                   rnntOptions& options,
                   std::vector<float>& num_grad) {
  float epsilon = 1e-2;

  for (size_t i = 0; i < num_grad.size(); ++i) {
    std::vector<float> costsP1(minibatch);
    std::vector<float> costsP2(minibatch);

    acts[i] += epsilon;
    throw_on_error(compute_rnnt_loss(acts.data(),
                                     NULL,
                                     flat_labels.data(),
                                     label_lengths.data(),
                                     sizes.data(),
                                     alphabet_size,
                                     minibatch,
                                     costsP1.data(),
                                     rnnt_cpu_workspace,
                                     options),
                   "Error: compute_rnnt_loss (1) in grad_check");

    acts[i] -= 2 * epsilon;
    throw_on_error(compute_rnnt_loss(acts.data(),
                                     NULL,
                                     flat_labels.data(),
                                     label_lengths.data(),
                                     sizes.data(),
                                     alphabet_size,
                                     minibatch,
                                     costsP2.data(),
                                     rnnt_cpu_workspace,
                                     options),
                   "Error: compute_rnnt_loss (2) in grad_check");

    float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.);
    float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.);

    acts[i] += epsilon;
    num_grad[i] = (costP1 - costP2) / (2 * epsilon);
  }
  return 0.0;
}

bool grad_check(int T,
                int L,
                int alphabet_size,
                std::vector<float>& acts,
                const std::vector<std::vector<int>>& labels,
                const std::vector<int>& sizes,
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
  options.loc = RNNT_CPU;
  options.num_threads = 1;
  options.batch_first = true;

  size_t cpu_alloc_bytes;
  throw_on_error(
      get_rnnt_workspace_size(T, L, sizes.size(), false, &cpu_alloc_bytes),
      "Error: get_rnnt_workspace_size in grad_check");

  void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

  throw_on_error(compute_rnnt_loss(acts.data(),
                                   grads.data(),
                                   flat_labels.data(),
                                   label_lengths.data(),
                                   sizes.data(),
                                   alphabet_size,
                                   minibatch,
                                   costs.data(),
                                   rnnt_cpu_workspace,
                                   options),
                 "Error: compute_rnnt_loss (0) in grad_check");
  float cost = std::accumulate(costs.begin(), costs.end(), 0.);
  cost = cost;

  std::vector<float> num_grad(grads.size());

  // perform 2nd order central differencing
  numeric_grad(acts,
               flat_labels,
               label_lengths,
               sizes,
               alphabet_size,
               minibatch,
               rnnt_cpu_workspace,
               options,
               num_grad);
  free(rnnt_cpu_workspace);

  float diff = rel_diff(grads, num_grad);
  return diff < tol;
}

bool run_tests() {
  std::vector<std::tuple<int, int, int, int, float>> problem_sizes = {
      std::make_tuple(20, 50, 15, 1, 1e-4),
      std::make_tuple(5, 10, 5, 65, 1e-4)};

  std::mt19937 gen(2);

  bool status = true;
  for (auto problem : problem_sizes) {
    int alphabet_size, T, L, minibatch;
    float tol;
    std::tie(alphabet_size, T, L, minibatch, tol) = problem;

    std::cout << "alphabet_size: " << alphabet_size << std::endl;
    std::cout << "T: " << T << std::endl;
    std::cout << "L: " << L << std::endl;
    std::cout << "minibatch: " << minibatch << std::endl;
    std::cout << "tol: " << tol << std::endl;

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

  std::cout << "Running CPU tests" << std::endl;

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
