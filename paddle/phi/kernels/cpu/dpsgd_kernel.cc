// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>
#include <stdlib.h>

#include <iostream>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void DpsgdOpKernel(const Context &dev_ctx,
                   const DenseTensor &param_in,
                   const DenseTensor &grad_in,
                   const DenseTensor &learning_rate_in,
                   float clip_in,
                   float batch_size_in,
                   float sigma_in,
                   int seed_in,
                   DenseTensor *param_out) {
  const auto *learning_rate = &learning_rate_in;

  const auto *param = &param_in;
  const auto *grad = &grad_in;

  auto sz = param_out->numel();
  PADDLE_ENFORCE_EQ(param->numel(),
                    sz,
                    common::errors::InvalidArgument(
                        "Input parameter's number of elements is error, "
                        "expected %zu, but received %zu."));
  PADDLE_ENFORCE_EQ(grad->numel(),
                    sz,
                    common::errors::InvalidArgument(
                        "Input gradient's number of elements is error, "
                        "expected %zu, but received %zu."));

  const T *lr = learning_rate->data<T>();
  const T *param_data = param->data<T>();
  const T *grad_data = grad->data<T>();

  T *out_data = dev_ctx.template Alloc<T>(param_out);

  T clip = static_cast<T>(clip_in);
  T batch_size = static_cast<T>(batch_size_in);
  T sigma = static_cast<T>(sigma_in);

  // compute clipping
  float l2_norm = 0.0;
  for (int64_t i = 0; i < grad->numel(); ++i) {
    l2_norm = l2_norm + grad_data[i] * grad_data[i];
  }
  l2_norm = std::sqrt(l2_norm);

  float scale = 1.0;
  if (l2_norm > clip) {
    scale = l2_norm / clip;
  }

  // generate gaussian noise.
  // [https://en.wikipedia.org/wiki/Box-Muller_transform]
  float V1, V2, S;
  float X;
  float mu = 0.0;
  float U1, U2;
  unsigned seed = static_cast<unsigned int>(seed_in);
  if (seed == 0) {
    seed = (unsigned)(time(NULL));
  }
  std::minstd_rand engine;
  engine.seed(seed);
  std::uniform_real_distribution<T> dist(0.0, 1.0);
  do {
    U1 = dist(engine);
    U2 = dist(engine);
    V1 = 2 * U1 - 1;
    V2 = 2 * U2 - 1;
    S = V1 * V1 + V2 * V2;
  } while (S >= 1 || S == 0);

  X = V1 * sqrt(-2 * log(S) / S);

  float gaussian_noise = mu + X * sigma;

  // update parameters
  for (int64_t i = 0; i < grad->numel(); ++i) {
    out_data[i] = param_data[i] -
                  lr[0] * (grad_data[i] / scale + gaussian_noise / batch_size);
  }
  // CCS16 - Deep Learning with Differential Privacy.
  // [https://arxiv.org/abs/1607.00133]
}  // Compute

}  // namespace phi

PD_REGISTER_KERNEL(dpsgd, CPU, ALL_LAYOUT, phi::DpsgdOpKernel, float, double) {}
