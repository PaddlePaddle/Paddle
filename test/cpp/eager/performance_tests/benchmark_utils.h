// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <math.h>

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/phi/api/all.h"

/* MLP Configurations */
// Out1 = X[M, N] x W[N, K] + B[K]
// ... x MLP_NUM_LINEAR
// Out  = ReduceSum(OutN)
#define MLP_M 4
#define MLP_N 16
#define MLP_K MLP_N
#define MLP_X_VAL 1.0
#define MLP_W_VAL 2.0
#define MLP_B_VAL 3.0
#define MLP_NUM_LINEAR 1000

namespace egr {

inline std::unordered_map<std::string, float> compute_mlp_expected_results() {
  float Out = MLP_X_VAL;
  for (size_t i = 0; i < MLP_NUM_LINEAR; i++) {
    Out = Out * MLP_W_VAL * MLP_N + MLP_B_VAL;
  }
  Out = Out * MLP_M * MLP_N;

  float GradX = 1.0 * pow((MLP_W_VAL * MLP_N), MLP_NUM_LINEAR);
  float GradW0 =
      1.0 * pow((MLP_W_VAL * MLP_N), (MLP_NUM_LINEAR - 1)) * MLP_X_VAL * MLP_M;
  return {{"Out", Out}, {"GradX", GradX}, {"GradW", GradW0}};
}

/* ---- Eager Scale ---- */
void benchmark_eager_scale(const paddle::Tensor& tensor,
                           bool accuracy_check = false);

/* ---- Eager MatMul ---- */
void benchmark_eager_matmul(const paddle::Tensor& X,
                            const paddle::Tensor& Y,
                            bool accuracy_check = false);

void benchmark_eager_intermediate_matmul(const paddle::Tensor& X,
                                         const paddle::Tensor& Y,
                                         bool accuracy_check = false);

void benchmark_eager_intermediate_mlp(const paddle::Tensor& X,
                                      const std::vector<paddle::Tensor>& Ws,
                                      const std::vector<paddle::Tensor>& Bs,
                                      bool accuracy_check = false);

}  // namespace egr

namespace paddle {
namespace imperative {
/* ---- Fluid Scale ---- */
// TODO(jiabin): Change this and remove nolint
void benchmark_fluid_scale(
    const std::shared_ptr<imperative::VarBase>& X,  // NOLINT
    const phi::Place& place,
    bool accuracy_check = false);

/* ---- Fluid MatMul ---- */
void benchmark_fluid_matmul(
    const std::shared_ptr<imperative::VarBase>& X,
    const std::shared_ptr<imperative::VarBase>& Y,  // NOLINT
    const phi::Place& place,
    bool accuracy_check = false);

/* ---- Fluid MLP ---- */
void benchmark_fluid_mlp(
    const std::shared_ptr<imperative::VarBase>& X,
    const std::vector<std::shared_ptr<imperative::VarBase>>& Ws,
    const std::vector<std::shared_ptr<imperative::VarBase>>& Bs,
    const phi::Place& place,
    bool accuracy_check = false);

}  // namespace imperative
}  // namespace paddle
